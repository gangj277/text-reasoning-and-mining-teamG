"""
robustness.py — 적대적 검토 대응: 중심 주장의 강건성 실험.

A) 시드단어 마스킹 ablation (핵심):
   config 시드 키워드를 골드 헤드라인에서 제거하고 재예측 → 외부 성능이 유지되면
   "전이는 시드 1:1 매칭이 아니라 분포적 학습"임을 입증. 붕괴하면 솔직히 보고.
B) 100% 실데이터 교차검증:
   B1) 손-라벨 골드 102건 5-fold stratified CV (실train/실test, 깨끗한 라벨)
   B2) 약-라벨 실뉴스 2062건 5-fold CV (대규모, 노이즈 라벨 → 하한)
   → 합성 의존 없는 실데이터 분류 가능성 수치.
D) 진짜 어휘 커버리지:
   |gold_vocab ∩ train_vocab| / |gold_vocab|, 헤드라인당 중앙 in-vocab 특징 수.
결과: results/robustness.json
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from config import (RESULTS_DIR, RISK_TYPES, GOLD_PATH, SEED, DATA_DIR)
from preprocess import load_jsonl, tokenize
from classify import load_xy, train_final_model, _vectorizer

SEED_WORDS = set()
for r in RISK_TYPES:
    SEED_WORDS.update(r["seeds_ko"]); SEED_WORDS.update(r["seeds_en"])


def _gold_tokens():
    gold = load_jsonl(GOLD_PATH)
    toks = [tokenize(g["title"], g["lang"]) for g in gold]
    y = np.array([g["gold"] for g in gold])
    lang = np.array([g["lang"] for g in gold])
    return gold, toks, y, lang


def ablation_seed_mask():
    """A) 시드단어를 골드에서 제거 후 합성학습 모델로 재예측."""
    X, ytr, _ = load_xy()
    vec, clf = train_final_model(X, ytr)
    _, toks, y, _ = _gold_tokens()
    full = vec.transform([" ".join(t) for t in toks])
    masked = vec.transform([" ".join(w for w in t if w not in SEED_WORDS) for t in toks])
    res = {}
    for name, M in (("full", full), ("seed_masked", masked)):
        p = clf.predict(M)
        res[name] = {"acc": round(float(accuracy_score(y, p)), 4),
                     "macro_f1": round(float(f1_score(y, p, average="macro")), 4)}
    # 마스킹으로 제거된 토큰 비율
    tot = sum(len(t) for t in toks)
    removed = sum(1 for t in toks for w in t if w in SEED_WORDS)
    res["pct_tokens_removed"] = round(removed / max(1, tot), 3)
    res["interpretation"] = ("시드 제거 후에도 성능이 대부분 유지되면 분포적 학습(강건), "
                             "붕괴하면 시드매칭 의존")
    return res


def cv_on(token_lists, y, k=5, label=""):
    X = np.array([" ".join(t) for t in token_lists], dtype=object)
    y = np.array(y)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    f1s, accs = [], []
    for tr, te in skf.split(X, y):
        vec = _vectorizer()
        Xtr = vec.fit_transform(X[tr]); Xte = vec.transform(X[te])
        clf = LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")
        clf.fit(Xtr, y[tr]); p = clf.predict(Xte)
        f1s.append(f1_score(y[te], p, average="macro")); accs.append(accuracy_score(y[te], p))
    return {"label": label, "k": k, "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "macro_f1_std": round(float(np.std(f1s)), 4),
            "acc_mean": round(float(np.mean(accs)), 4), "n": int(len(y))}


def real_cv():
    """B1) 골드 CV, B2) 약-라벨 실뉴스 CV."""
    _, gtoks, gy, _ = _gold_tokens()
    b1 = cv_on(gtoks, gy, k=5, label="gold_handlabeled_102")

    real = load_jsonl(os.path.join(DATA_DIR, "realnews_raw.jsonl"))
    rtoks = [tokenize(r["title"], r["lang"]) for r in real]
    ry = [r["query_type"] for r in real]
    b2 = cv_on(rtoks, ry, k=5, label="realnews_weaklabel_2062")
    return {"gold_cv": b1, "weaklabel_cv": b2}


def vocab_metrics():
    X, y, _ = load_xy()
    vec, _ = train_final_model(X, y)
    train_vocab = set(vec.vocabulary_.keys())
    # unigram 어휘만 비교(2gram 제외)
    train_uni = {w for w in train_vocab if " " not in w}
    _, gtoks, _, _ = _gold_tokens()
    gold_vocab = set(w for t in gtoks for w in t)
    inter = gold_vocab & train_uni
    coverage = len(inter) / max(1, len(gold_vocab))
    feats = vec.transform([" ".join(t) for t in gtoks])
    per_doc = np.asarray((feats > 0).sum(axis=1)).ravel()
    return {"gold_vocab_size": len(gold_vocab),
            "type_token_coverage": round(coverage, 3),
            "median_invocab_features_per_headline": int(np.median(per_doc)),
            "min": int(per_doc.min()), "max": int(per_doc.max()),
            "note": "type_token_coverage = |gold∩train unigram| / |gold|  (기존 0.98 은 OOV-아닌-문서비율로 오표기였음)"}


def main():
    out = {"A_seed_mask_ablation": ablation_seed_mask(),
           "B_real_cv": real_cv(),
           "D_vocab_coverage": vocab_metrics()}
    json.dump(out, open(os.path.join(RESULTS_DIR, "robustness.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    o = main()
    a = o["A_seed_mask_ablation"]
    print("=== A) 시드단어 마스킹 ablation (중심주장 검증) ===")
    print(f"   full        acc={a['full']['acc']}  F1={a['full']['macro_f1']}")
    print(f"   seed_masked acc={a['seed_masked']['acc']}  F1={a['seed_masked']['macro_f1']}  "
          f"(제거토큰 {a['pct_tokens_removed']*100:.0f}%)")
    print("=== B) 100% 실데이터 교차검증 ===")
    print(f"   B1 골드 손-라벨 CV:  F1={o['B_real_cv']['gold_cv']['macro_f1_mean']}"
          f"±{o['B_real_cv']['gold_cv']['macro_f1_std']} (n={o['B_real_cv']['gold_cv']['n']})")
    print(f"   B2 약-라벨 실뉴스 CV: F1={o['B_real_cv']['weaklabel_cv']['macro_f1_mean']}"
          f"±{o['B_real_cv']['weaklabel_cv']['macro_f1_std']} (n={o['B_real_cv']['weaklabel_cv']['n']})")
    print("=== D) 진짜 어휘 커버리지 ===")
    print(f"   type-token coverage={o['D_vocab_coverage']['type_token_coverage']} | "
          f"median in-vocab feats/headline={o['D_vocab_coverage']['median_invocab_features_per_headline']}")
