"""
stats_addendum.py — 통계적 엄밀성 보강 (적대적 검토 선제 대응).

외부 골드셋(n=102) 결과에 대해:
 - 부트스트랩 95% CI (정확도, Macro-F1)
 - 다수클래스(majority) 베이스라인
 - KO vs EN 정확도 차이의 부트스트랩 CI (유의성)
 - Wilson 정확도 구간
결과: results/stats_addendum.json
"""
from __future__ import annotations
import json, os
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score

from config import RESULTS_DIR, GOLD_PATH, SEED
from preprocess import load_jsonl, tokenize
from classify import load_xy, train_final_model


def predict_gold():
    X, y, _ = load_xy()
    vec, clf = train_final_model(X, y)
    gold = load_jsonl(GOLD_PATH)
    Xg = vec.transform([" ".join(tokenize(g["title"], g["lang"])) for g in gold])
    yt = np.array([g["gold"] for g in gold])
    yp = clf.predict(Xg)
    lang = np.array([g["lang"] for g in gold])
    return yt, yp, lang


def bootstrap_ci(yt, yp, B=3000, metric="acc"):
    rng = np.random.RandomState(SEED)
    n = len(yt); stats = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if metric == "acc":
            stats.append(accuracy_score(yt[idx], yp[idx]))
        else:
            stats.append(f1_score(yt[idx], yp[idx], average="macro"))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def wilson(p, n, z=1.96):
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return round(float(center-half), 4), round(float(center+half), 4)


def main():
    yt, yp, lang = predict_gold()
    acc = accuracy_score(yt, yp); macro = f1_score(yt, yp, average="macro")

    acc_ci = bootstrap_ci(yt, yp, metric="acc")
    f1_ci = bootstrap_ci(yt, yp, metric="f1")
    wil = wilson(acc, len(yt))

    # majority baseline
    maj = Counter(yt).most_common(1)[0][0]
    maj_acc = float((yt == maj).mean())

    # KO vs EN diff bootstrap CI
    rng = np.random.RandomState(SEED)
    ko_i = np.where(lang == "ko")[0]; en_i = np.where(lang == "en")[0]
    diffs = []
    for _ in range(3000):
        ks = ko_i[rng.randint(0, len(ko_i), len(ko_i))]
        es = en_i[rng.randint(0, len(en_i), len(en_i))]
        diffs.append(accuracy_score(yt[ks], yp[ks]) - accuracy_score(yt[es], yp[es]))
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])

    out = {
        "n_gold": int(len(yt)),
        "external_acc": round(float(acc), 4), "acc_boot_ci95": acc_ci, "acc_wilson_ci95": wil,
        "external_macro_f1": round(float(macro), 4), "macro_f1_boot_ci95": f1_ci,
        "majority_baseline_acc": round(maj_acc, 4), "majority_class": maj,
        "random_baseline_acc": round(1/7, 4),
        "ko_minus_en_acc_diff": round(float(np.mean(diffs)), 4),
        "ko_minus_en_ci95": [round(float(d_lo), 4), round(float(d_hi), 4)],
        "ko_en_diff_significant": not (d_lo <= 0 <= d_hi),
    }
    json.dump(out, open(os.path.join(RESULTS_DIR, "stats_addendum.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    o = main()
    print("=== 통계 보강 (부트스트랩 95% CI) ===")
    print(f"  외부 정확도 {o['external_acc']}  95%CI {o['acc_boot_ci95']}  (Wilson {o['acc_wilson_ci95']})")
    print(f"  외부 Macro-F1 {o['external_macro_f1']}  95%CI {o['macro_f1_boot_ci95']}")
    print(f"  다수클래스 베이스라인 {o['majority_baseline_acc']} (={o['majority_class']}) | 무작위 {o['random_baseline_acc']}")
    print(f"  KO−EN 정확도 차이 {o['ko_minus_en_acc_diff']}  95%CI {o['ko_minus_en_ci95']}  "
          f"유의={o['ko_en_diff_significant']}")
