"""
neutral_eval.py — label-blind 외부검증 (순환성 근본 해소).

적대적 검토 핵심: 기존 골드는 '타입별 질의'로 수집돼 라벨=질의키워드(query-echo 0.99).
해소: '중립 질의'("공급망"/"supply chain")로 수집한 혼합 스트림에서, 질의힌트 없이
연구자가 7유형으로 분류 가능한 건만 blind 라벨링 → 라벨이 질의로 결정되지 않음.
이 골드에서 query-echo 는 무의미하므로, majority 대비 우월하면 '진짜 일반화'다.
결과: results/neutral_eval.json
"""
from __future__ import annotations
import json, os
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score

from config import RESULTS_DIR, DATA_DIR, RISK_CODES, CODE2KO
from preprocess import tokenize
from classify import load_xy, train_final_model
from robustness import SEED_WORDS

# 중립질의 100건에 대한 blind 라벨 (7유형 명확분류만; 전략/시상/오피니언/사이버 등 74건 DROP)
LABELS = {
    0: "SUPPLIER", 5: "SUPPLIER", 9: "SUPPLIER", 13: "FINANCIAL", 17: "LOGISTICS",
    22: "LABOR", 23: "LOGISTICS", 27: "GEOPOLITICAL", 30: "NATURAL_DISASTER",
    32: "GEOPOLITICAL", 33: "SUPPLIER", 38: "LABOR", 45: "LABOR", 48: "SUPPLIER",
    49: "FINANCIAL", 51: "GEOPOLITICAL", 53: "FINANCIAL", 63: "GEOPOLITICAL",
    67: "SUPPLIER", 72: "LOGISTICS", 78: "GEOPOLITICAL", 80: "LOGISTICS",
    81: "GEOPOLITICAL", 83: "GEOPOLITICAL", 87: "GEOPOLITICAL", 93: "SUPPLIER",
}


def main():
    sample = json.load(open(os.path.join(DATA_DIR, "neutral_sample.json"), encoding="utf-8"))
    gold = [{"title": sample[i]["title"], "lang": sample[i]["lang"], "gold": c}
            for i, c in LABELS.items()]
    json.dump(gold, open(os.path.join(DATA_DIR, "neutral_gold.jsonl"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    X, y, _ = load_xy()
    vec, clf = train_final_model(X, y)
    toks = [tokenize(g["title"], g["lang"]) for g in gold]
    yt = np.array([g["gold"] for g in gold])

    full = vec.transform([" ".join(t) for t in toks])
    masked = vec.transform([" ".join(w for w in t if w not in SEED_WORDS) for t in toks])
    p_full = clf.predict(full); p_mask = clf.predict(masked)

    present = [c for c in RISK_CODES if (yt == c).any()]
    maj = Counter(yt).most_common(1)[0]
    out = {
        "n": len(gold), "n_dropped_of_100": 100 - len(gold),
        "lang_dist": dict(Counter(g["lang"] for g in gold)),
        "class_dist": dict(Counter(yt.tolist())),
        "full": {"acc": round(float(accuracy_score(yt, p_full)), 4),
                 "macro_f1_present": round(float(f1_score(yt, p_full, average="macro", labels=present)), 4)},
        "seed_masked": {"acc": round(float(accuracy_score(yt, p_mask)), 4),
                        "macro_f1_present": round(float(f1_score(yt, p_mask, average="macro", labels=present)), 4)},
        "majority_baseline_acc": round(maj[1] / len(gold), 4),
        "majority_class": maj[0], "random_acc": round(1/7, 4),
        "note": ("중립질의 label-blind 골드. query-echo 무의미(단일 일반질의). "
                 "full acc 가 majority 를 상회하면 질의복원이 아닌 진짜 일반화."),
    }
    json.dump(out, open(os.path.join(RESULTS_DIR, "neutral_eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # 오분류 정성
    errs = [(gold[i]["title"][:55], CODE2KO[yt[i]], CODE2KO[p_full[i]])
            for i in range(len(gold)) if yt[i] != p_full[i]]
    return out, errs


if __name__ == "__main__":
    o, errs = main()
    print("=== label-blind 중립질의 외부검증 (순환성 근본 해소) ===")
    print(f"  n={o['n']} (100건 중 {o['n_dropped_of_100']} drop), 분포={o['class_dist']}")
    print(f"  full         acc={o['full']['acc']}  macroF1(present)={o['full']['macro_f1_present']}")
    print(f"  seed_masked  acc={o['seed_masked']['acc']}  macroF1(present)={o['seed_masked']['macro_f1_present']}")
    print(f"  majority     acc={o['majority_baseline_acc']} ({CODE2KO[o['majority_class']]}) | random {o['random_acc']}")
    print(f"  → full acc {o['full']['acc']} vs majority {o['majority_baseline_acc']}: "
          f"{o['full']['acc']/o['majority_baseline_acc']:.2f}배")
    print(f"\n  오분류 {len(errs)}건:")
    for t, g, p in errs[:8]:
        print(f"   gold={g} / pred={p} | {t}")
