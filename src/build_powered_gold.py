"""
build_powered_gold.py — 검정력 있는 label-blind 골드 구축 + 평가 (자기리뷰 후속).

입력: data/panel_annotations.json  = {"A":{idx:label}, "B":{...}, "C":{...}}  (3인 패널)
       data/anno_batch.json         = [{i, lang, title}, ...]  (중립질의 700건)
절차:
 1) Fleiss κ (3인 평정자 신뢰도) + 라벨 매트릭스
 2) 합의 골드: 다수결(≥2/3), OTHER·무합의 제외  → 순환 없는 powered gold
 3) 평가:
    a. 합성 학습 모델 전이 (full / seed-masked) + 부트스트랩 CI + majority 대비 유의성
    b. real-train/real-test 5-fold CV (완전히 깨끗한 신규 실험)
결과: data/powered_gold.jsonl, results/powered_eval.json
"""
from __future__ import annotations
import json, os
import numpy as np
from collections import Counter
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from config import RESULTS_DIR, DATA_DIR, RISK_CODES, SEED, CODE2KO
from preprocess import tokenize
from classify import load_xy, train_final_model, _vectorizer
from robustness import SEED_WORDS

CATS = RISK_CODES + ["OTHER"]


def fleiss_kappa(matrix):
    """matrix: N x k counts (each row sums to n raters)."""
    N, k = matrix.shape
    n = matrix[0].sum()
    p_j = matrix.sum(axis=0) / (N * n)
    P_i = (np.square(matrix).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean(); P_e = np.square(p_j).sum()
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else 0.0


def cohen(a, b):
    items = [i for i in a if i in b]
    ya = [a[i] for i in items]; yb = [b[i] for i in items]
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(ya, yb))


def boot_ci(correct, B=10000):
    n = len(correct); rng = np.random.RandomState(SEED)
    s = [correct[rng.randint(0, n, n)].mean() for _ in range(B)]
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def main():
    ann = json.load(open(os.path.join(DATA_DIR, "panel_annotations.json"), encoding="utf-8"))
    batch = json.load(open(os.path.join(DATA_DIR, "anno_batch.json"), encoding="utf-8"))
    by_i = {b["i"]: b for b in batch}
    A, Bn, C = ann["A"], ann["B"], ann["C"]
    idxs = sorted(set(int(i) for i in A) & set(int(i) for i in Bn) & set(int(i) for i in C))

    # Fleiss matrix
    cat_idx = {c: j for j, c in enumerate(CATS)}
    M = np.zeros((len(idxs), len(CATS)), dtype=int)
    labels3 = {}
    for r, i in enumerate(idxs):
        votes = [A[str(i)], Bn[str(i)], C[str(i)]]
        labels3[i] = votes
        for v in votes:
            M[r, cat_idx[v]] += 1
    kappa = fleiss_kappa(M)
    pair = {"A-B": cohen(A, Bn), "A-C": cohen(A, C), "B-C": cohen(Bn, C)}

    # consensus gold: majority >=2, not OTHER, not all-different
    gold = []
    for i in idxs:
        c = Counter(labels3[i]); lab, ct = c.most_common(1)[0]
        if ct >= 2 and lab != "OTHER":
            gold.append({"title": by_i[i]["title"], "lang": by_i[i]["lang"], "gold": lab})
    with open(os.path.join(DATA_DIR, "powered_gold.jsonl"), "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    yt = np.array([g["gold"] for g in gold]); langs = np.array([g["lang"] for g in gold])
    toks = [tokenize(g["title"], g["lang"]) for g in gold]
    n = len(gold)
    maj_cls, maj_n = Counter(yt).most_common(1)[0]; maj = maj_n / n

    # (a) 합성 학습 모델 전이
    X, y, _ = load_xy(); vec, clf = train_final_model(X, y)
    full = clf.predict(vec.transform([" ".join(t) for t in toks]))
    mask = clf.predict(vec.transform([" ".join(w for w in t if w not in SEED_WORDS) for t in toks]))
    c_full = (full == yt).astype(int); c_mask = (mask == yt).astype(int)
    acc_full = c_full.mean()
    p_vs_maj = stats.binomtest(int(c_full.sum()), n, maj, alternative="greater").pvalue
    p_vs_rand = stats.binomtest(int(c_full.sum()), n, 1/7, alternative="greater").pvalue
    per_lang = {}
    for lg in ("ko", "en"):
        m = langs == lg
        if m.sum():
            per_lang[lg] = {"n": int(m.sum()), "acc": round(float((full[m] == yt[m]).mean()), 4)}

    # (b) real-train/real-test 5-fold CV (완전히 깨끗)
    Xr = np.array([" ".join(t) for t in toks], dtype=object)
    cv_f1, cv_acc = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(Xr, yt):
        v = _vectorizer(); Xt = v.fit_transform(Xr[tr]); Xe = v.transform(Xr[te])
        m = LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced").fit(Xt, yt[tr])
        pr = m.predict(Xe)
        cv_f1.append(f1_score(yt[te], pr, average="macro")); cv_acc.append(accuracy_score(yt[te], pr))

    out = {
        "annotation": {"n_batch": len(idxs), "fleiss_kappa": round(float(kappa), 4),
                       "pairwise_cohen": {k: round(v, 4) for k, v in pair.items()},
                       "consensus_kept": n, "dropped_other_or_nomajority": len(idxs) - n},
        "gold": {"n": n, "class_dist": dict(Counter(yt.tolist())),
                 "lang_dist": dict(Counter(langs.tolist())),
                 "majority_class": maj_cls, "majority_acc": round(maj, 4)},
        "transfer_synthetic_to_real": {
            "full_acc": round(float(acc_full), 4),
            "full_macro_f1": round(float(f1_score(yt, full, average="macro")), 4),
            "full_acc_ci95": boot_ci(c_full),
            "seed_masked_acc": round(float(c_mask.mean()), 4),
            "p_vs_majority": round(float(p_vs_maj), 4), "sig_vs_majority": bool(p_vs_maj < 0.05),
            "p_vs_random": round(float(p_vs_rand), 5), "sig_vs_random": bool(p_vs_rand < 0.05),
            "per_lang": per_lang},
        "real_cv_on_powered_gold": {
            "macro_f1_mean": round(float(np.mean(cv_f1)), 4), "macro_f1_std": round(float(np.std(cv_f1)), 4),
            "acc_mean": round(float(np.mean(cv_acc)), 4)},
    }
    json.dump(out, open(os.path.join(RESULTS_DIR, "powered_eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    o = main()
    a = o["annotation"]; g = o["gold"]; t = o["transfer_synthetic_to_real"]; cv = o["real_cv_on_powered_gold"]
    print(f"=== 패널 신뢰도 ===  Fleiss κ={a['fleiss_kappa']}  pairwise={a['pairwise_cohen']}")
    print(f"  합의 골드 {g['n']}건 (배치 {a['n_batch']} 중 OTHER/무합의 {a['dropped_other_or_nomajority']} 제외)")
    print(f"  분포 {g['class_dist']} | 언어 {g['lang_dist']} | majority={g['majority_acc']} ({CODE2KO.get(g['majority_class'],'')})")
    print(f"\n=== (a) 합성→실 전이 (powered) ===")
    print(f"  full acc={t['full_acc']}  95%CI {t['full_acc_ci95']}  macroF1={t['full_macro_f1']}")
    print(f"  seed-masked acc={t['seed_masked_acc']}")
    print(f"  vs majority: p={t['p_vs_majority']}  유의={t['sig_vs_majority']}")
    print(f"  vs random:   p={t['p_vs_random']}  유의={t['sig_vs_random']}")
    print(f"  언어별 {t['per_lang']}")
    print(f"\n=== (b) real-train/real-test CV (깨끗) ===")
    print(f"  Macro-F1={cv['macro_f1_mean']}±{cv['macro_f1_std']}  acc={cv['acc_mean']}")
