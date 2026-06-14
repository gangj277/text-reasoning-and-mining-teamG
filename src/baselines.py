"""
baselines.py — 정직한 베이스라인 비교 (적대적 검증 must-fix).

적대적 검토가 지적한 핵심: 골드가 타입별 질의로 수집돼 '라벨=질의키워드'에 가깝다.
이를 드러내기 위해 trivial/경쟁 베이스라인을 실측한다:
 - query_echo : 라벨 = 검색질의(src_hint). ML 없음. (순환성의 상한을 폭로)
 - seed_rule  : 원문에서 시드 키워드 substring 매칭 argmax. 전처리/ML 없음.
 - char_ngram : 전처리 없이 char n-gram TF-IDF + LR. (KoNLPy/NLTK 불필요)
 - full       : 본 파이프라인(참고용, external_eval 결과)
결과: results/baselines.json
"""
from __future__ import annotations
import json, os
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from config import RESULTS_DIR, RISK_TYPES, RISK_CODES, GOLD_PATH, SEED, CORPUS_PATH
from preprocess import load_jsonl

SEED_BY_TYPE = {r["code"]: [s.lower() for s in (r["seeds_ko"] + r["seeds_en"])] for r in RISK_TYPES}


def query_echo(gold):
    yt = [g["gold"] for g in gold]; yp = [g.get("src_hint", "?") for g in gold]
    return {"macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
            "acc": round(float(accuracy_score(yt, yp)), 4),
            "label_eq_hint_rate": round(float(np.mean([a == b for a, b in zip(yt, yp)])), 3)}


def seed_rule(gold):
    yt, yp = [], []
    for g in gold:
        t = g["title"].lower()
        scores = {c: sum(t.count(s) for s in seeds) for c, seeds in SEED_BY_TYPE.items()}
        best = max(scores, key=scores.get)
        yp.append(best if scores[best] > 0 else "NONE")
        yt.append(g["gold"])
    cov = np.mean([p != "NONE" for p in yp])
    idx = [i for i in range(len(yt)) if yp[i] != "NONE"]
    return {"macro_f1_covered": round(float(f1_score([yt[i] for i in idx], [yp[i] for i in idx],
            average="macro")), 4) if idx else 0.0,
            "acc_all": round(float(accuracy_score(yt, yp)), 4), "coverage": round(float(cov), 3)}


def char_ngram(gold):
    docs = load_jsonl(CORPUS_PATH)
    Xtr = [d["text"] for d in docs]; ytr = np.array([d["risk_type"] for d in docs])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    Xtr_v = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced").fit(Xtr_v, ytr)
    yt = np.array([g["gold"] for g in gold]); Xg = vec.transform([g["title"] for g in gold])
    yp = clf.predict(Xg)
    return {"macro_f1": round(float(f1_score(yt, yp, average="macro")), 4),
            "acc": round(float(accuracy_score(yt, yp)), 4)}


def main():
    gold = load_jsonl(GOLD_PATH)
    ext = json.load(open(os.path.join(RESULTS_DIR, "rq_external.json"), encoding="utf-8"))
    out = {
        "query_echo": query_echo(gold),
        "seed_rule": seed_rule(gold),
        "char_ngram_no_preproc": char_ngram(gold),
        "full_pipeline": {"macro_f1": ext["external_macro_f1"], "acc": ext["external_acc"]},
        "majority": {"acc": round(max(Counter(g["gold"] for g in gold).values())/len(gold), 4)},
        "random": {"acc": round(1/len(RISK_CODES), 4)},
        "interpretation": ("query_echo(0.99)>full(0.77)>seed_rule≈char_ngram(~0.72)>majority(0.17). "
                           "타입질의 골드에서는 trivial echo가 최고 → 외부수치는 '질의버킷 복원'이며 "
                           "본 방법의 이득은 modest. 진짜 일반화는 neutral_gold 로 별도 검증."),
    }
    json.dump(out, open(os.path.join(RESULTS_DIR, "baselines.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    o = main()
    print("=== 정직한 베이스라인 비교 (타입질의 골드 n=102) ===")
    print(f"  query_echo (라벨=질의):   F1={o['query_echo']['macro_f1']}  acc={o['query_echo']['acc']}"
          f"  (라벨=힌트 {o['query_echo']['label_eq_hint_rate']})")
    print(f"  full pipeline:            F1={o['full_pipeline']['macro_f1']}  acc={o['full_pipeline']['acc']}")
    print(f"  seed_rule (키워드규칙):   F1={o['seed_rule']['macro_f1_covered']}  acc={o['seed_rule']['acc_all']}")
    print(f"  char_ngram (전처리X):     F1={o['char_ngram_no_preproc']['macro_f1']}  acc={o['char_ngram_no_preproc']['acc']}")
    print(f"  majority / random:        acc={o['majority']['acc']} / {o['random']['acc']}")
