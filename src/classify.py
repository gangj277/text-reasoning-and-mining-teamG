"""
classify.py — RQ2: 임베딩 확장 + 분류기가 리스크 유형을 정확히 분류하는가?
              RQ3: 단일 파이프라인이 한국어/영어를 일관되게 처리하는가?

 - 특징: TF-IDF (1~2gram), 전처리된 토큰 기반.
 - 분류기: LogisticRegression(화이트박스, 계수 해석가능) — 주모델.
           NB/LinearSVC 와 비교(ablation).
 - 평가: 층화 held-out 테스트, Macro-F1.
 - 학습곡선: 학습데이터 규모(16~800) vs Macro-F1 (3회 반복 평균) → 임계값 돌파 지점.
 - 언어별: 단일 모델의 KO/EN 테스트 성능 분리 측정(RQ3).
결과: results/rq2_classify.json, results/tables/classification_report.csv
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import SEED, RESULTS_DIR, TABLE_DIR, RISK_CODES, CORPUS_PATH
from preprocess import load_jsonl, tokenize_docs


def _vectorizer():
    return TfidfVectorizer(tokenizer=str.split, preprocessor=None, lowercase=False,
                           token_pattern=None, min_df=2, ngram_range=(1, 2), sublinear_tf=True)


def load_xy():
    docs = tokenize_docs(load_jsonl(CORPUS_PATH))
    X = [" ".join(d["tokens"]) for d in docs]
    y = np.array([d["risk_type"] for d in docs])
    lang = np.array([d["lang"] for d in docs])
    return X, y, lang


def train_final_model(X, y):
    """전체 학습데이터로 최종 모델(벡터라이저+분류기) 학습 → 외부 골드셋 평가에 재사용."""
    vec = _vectorizer(); Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")
    clf.fit(Xv, y)
    return vec, clf


def run_rq2_rq3():
    X, y, lang = load_xy()
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=y)
    Xtr = [X[i] for i in tr]; Xte = [X[i] for i in te]
    ytr, yte = y[tr], y[te]; lang_te = lang[te]

    vec = _vectorizer(); Xtr_v = vec.fit_transform(Xtr); Xte_v = vec.transform(Xte)
    print(f"  학습 {len(tr)} / 테스트 {len(te)}  | 특징 {Xtr_v.shape[1]}")

    # ----- 분류기 비교(ablation) -----
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced"),
        "LinearSVC": LinearSVC(C=1.0, class_weight="balanced"),
        "MultinomialNB": MultinomialNB(alpha=0.1),
    }
    ablation = {}
    best_name, best_f1, best_clf = None, -1, None
    for name, m in models.items():
        m.fit(Xtr_v, ytr)
        f1 = f1_score(yte, m.predict(Xte_v), average="macro")
        ablation[name] = round(float(f1), 4)
        print(f"  {name:20s} macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_name, best_f1, best_clf = name, f1, m

    yhat = best_clf.predict(Xte_v)
    macro_f1 = float(f1_score(yte, yhat, average="macro"))

    # ----- RQ3: 언어별 성능 -----
    per_lang = {}
    for lg in ("ko", "en"):
        mask = lang_te == lg
        if mask.sum() > 0:
            per_lang[lg] = round(float(f1_score(yte[mask], yhat[mask], average="macro")), 4)
    print(f"  RQ3 언어별 macro-F1: {per_lang}")

    # ----- 학습곡선 (LogisticRegression 고정) -----
    sizes = [s for s in [16, 32, 64, 128, 256, 512, 800] if s <= len(tr)]
    curve = []
    rng = np.random.RandomState(SEED)
    for n in sizes:
        f1s = []
        for rep in range(3):
            # 층화 서브샘플
            sub = []
            per_c = max(1, n // len(set(ytr)))
            for c in set(ytr):
                pool = [i for i in range(len(ytr)) if ytr[i] == c]
                rng.shuffle(pool); sub += pool[:per_c]
            sub = sub[:n]
            m = LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")
            m.fit(Xtr_v[sub], ytr[sub])
            f1s.append(f1_score(yte, m.predict(Xte_v), average="macro"))
        curve.append({"n": n, "f1_mean": round(float(np.mean(f1s)), 4),
                      "f1_std": round(float(np.std(f1s)), 4)})
        print(f"  n={n:4d}  macro-F1={np.mean(f1s):.4f} ±{np.std(f1s):.3f}")
    thr = next((c["n"] for c in curve if c["f1_mean"] >= 0.6), None)

    # ----- 혼동행렬 + 리포트 -----
    cm = confusion_matrix(yte, yhat, labels=RISK_CODES).tolist()
    rep = classification_report(yte, yhat, labels=RISK_CODES, output_dict=True, zero_division=0)

    out = {"best_model": best_name, "ablation": ablation, "test_macro_f1": round(macro_f1, 4),
           "per_lang_f1": per_lang, "learning_curve": curve, "threshold_n_at_0.6": thr,
           "n_train": len(tr), "n_test": len(te), "n_features": int(Xtr_v.shape[1]),
           "confusion": {"labels": RISK_CODES, "matrix": cm},
           "per_class_f1": {c: round(rep[c]["f1-score"], 3) for c in RISK_CODES if c in rep}}
    with open(os.path.join(RESULTS_DIR, "rq2_classify.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    import csv
    with open(os.path.join(TABLE_DIR, "classification_report.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["risk_type", "precision", "recall", "f1", "support"])
        for c in RISK_CODES:
            if c in rep:
                w.writerow([c, round(rep[c]["precision"], 3), round(rep[c]["recall"], 3),
                            round(rep[c]["f1-score"], 3), int(rep[c]["support"])])
    return out


if __name__ == "__main__":
    print("=== RQ2/RQ3: 분류 + 학습곡선 + 언어별 ===")
    res = run_rq2_rq3()
    print(f"\n최적={res['best_model']}  test Macro-F1={res['test_macro_f1']}")
    print(f"임계값(F1≥0.6) 돌파 데이터 규모: {res['threshold_n_at_0.6']}")
