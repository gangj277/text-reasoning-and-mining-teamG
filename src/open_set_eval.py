"""
open_set_eval.py — final methodological upgrade: open-set SCRM triage.

The previous powered gold kept only the 217 consensus risk headlines from a
700-item neutral-query stream. This script keeps consensus OTHER labels too and
tests the more deployment-like question:

  H1. A 7-class model without an OTHER option is not a deployable triage model.
  H2. Explicit open-set training can learn relevance rejection, but the remaining
      closed-set risk typing stays modest under severe class imbalance.
  H3. Confidence-thresholding a synthetic 7-class model is at best a diagnostic
      upper bound, not a substitute for real open-set labels.

Outputs:
  data/open_gold.jsonl
  results/open_set_eval.json
  results/tables/open_set_oof_predictions.csv
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from classify import _vectorizer, load_xy, train_final_model
from config import DATA_DIR, RESULTS_DIR, RISK_CODES, SEED, TABLE_DIR
from preprocess import tokenize

OTHER = "OTHER"
OPEN_LABELS = RISK_CODES + [OTHER]


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_open_gold():
    """Consensus labels over the full neutral-query annotation batch."""
    ann = _read_json(os.path.join(DATA_DIR, "panel_annotations.json"))
    batch = _read_json(os.path.join(DATA_DIR, "anno_batch.json"))
    by_i = {int(b["i"]): b for b in batch}
    idxs = sorted(set(map(int, ann["A"])) & set(map(int, ann["B"])) & set(map(int, ann["C"])))

    gold, no_consensus = [], []
    for i in idxs:
        votes = [ann["A"][str(i)], ann["B"][str(i)], ann["C"][str(i)]]
        lab, ct = Counter(votes).most_common(1)[0]
        row = by_i[i]
        if ct >= 2:
            gold.append({
                "i": i,
                "title": row["title"],
                "lang": row["lang"],
                "gold": lab,
                "is_risk": lab != OTHER,
                "votes": votes,
            })
        else:
            no_consensus.append({"i": i, "title": row["title"], "lang": row["lang"], "votes": votes})

    with open(os.path.join(DATA_DIR, "open_gold.jsonl"), "w", encoding="utf-8") as f:
        for row in gold:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return gold, no_consensus


def _texts_labels(rows):
    toks = [tokenize(r["title"], r["lang"]) for r in rows]
    X = np.array([" ".join(t) for t in toks], dtype=object)
    y = np.array([r["gold"] for r in rows], dtype=object)
    langs = np.array([r["lang"] for r in rows], dtype=object)
    return X, y, langs


def _risk_binary(y):
    y = np.asarray(y, dtype=object)
    return (y != OTHER).astype(int)


def _safe_auc(y_true_bin, score):
    try:
        return round(float(roc_auc_score(y_true_bin, score)), 4)
    except Exception:
        return None


def _safe_ap(y_true_bin, score):
    try:
        return round(float(average_precision_score(y_true_bin, score)), 4)
    except Exception:
        return None


def _metric_pack(y_true, y_pred, risk_score=None):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    y_true_bin = _risk_binary(y_true)
    y_pred_bin = _risk_binary(y_pred)
    present_open_labels = [lab for lab in OPEN_LABELS if np.any(y_true == lab)]
    present_risk_labels = [lab for lab in RISK_CODES if np.any(y_true == lab)]
    p, r, f, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )
    risk_mask = y_true != OTHER
    return {
        "accuracy_8way": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1_present_labels": round(
            float(f1_score(y_true, y_pred, labels=present_open_labels, average="macro", zero_division=0)), 4
        ),
        "macro_f1_full_taxonomy": round(
            float(f1_score(y_true, y_pred, labels=OPEN_LABELS, average="macro", zero_division=0)), 4
        ),
        "risk_detection": {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "roc_auc": _safe_auc(y_true_bin, risk_score) if risk_score is not None else None,
            "average_precision": _safe_ap(y_true_bin, risk_score) if risk_score is not None else None,
        },
        "risk_type_macro_f1_present_risk_labels": round(
            float(f1_score(y_true[risk_mask], y_pred[risk_mask], labels=present_risk_labels, average="macro", zero_division=0)), 4
        ) if risk_mask.any() else 0.0,
        "risk_type_macro_f1_full_taxonomy": round(
            float(f1_score(y_true[risk_mask], y_pred[risk_mask], labels=RISK_CODES, average="macro", zero_division=0)), 4
        ) if risk_mask.any() else 0.0,
        "per_class_f1": {
            lab: round(float(v), 4)
            for lab, v in zip(
                OPEN_LABELS,
                f1_score(y_true, y_pred, labels=OPEN_LABELS, average=None, zero_division=0),
            )
        },
        "confusion": {
            "labels": OPEN_LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=OPEN_LABELS).tolist(),
        },
    }


def _bootstrap_ci(y_true, y_pred, metric, B=5000):
    rng = np.random.RandomState(SEED)
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        vals.append(metric(y_true[idx], y_pred[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def _add_ci(pack, y_true, y_pred):
    pack["ci95"] = {
        "accuracy_8way": _bootstrap_ci(y_true, y_pred, lambda a, b: accuracy_score(a, b)),
        "macro_f1_present_labels": _bootstrap_ci(
            y_true,
            y_pred,
            lambda a, b: f1_score(
                a,
                b,
                labels=[lab for lab in OPEN_LABELS if np.any(a == lab)],
                average="macro",
                zero_division=0,
            ),
        ),
        "risk_detection_f1": _bootstrap_ci(
            y_true,
            y_pred,
            lambda a, b: f1_score(_risk_binary(a), _risk_binary(b), zero_division=0),
        ),
    }
    return pack


def baselines(y):
    y = np.asarray(y, dtype=object)
    pred_other = np.array([OTHER] * len(y), dtype=object)
    pred_geo = np.array(["GEOPOLITICAL"] * len(y), dtype=object)
    pred_random = np.array(RISK_CODES, dtype=object)[np.arange(len(y)) % len(RISK_CODES)]
    return {
        "always_other": _metric_pack(y, pred_other, risk_score=np.zeros(len(y))),
        "always_geopolitical": _metric_pack(y, pred_geo, risk_score=np.ones(len(y))),
        "round_robin_risk_only": _metric_pack(y, pred_random, risk_score=np.ones(len(y))),
    }


def synthetic_forced_and_threshold(X, y):
    """Synthetic 7-class transfer against the full open gold."""
    Xs, ys, _ = load_xy()
    vec, clf = train_final_model(Xs, ys)
    proba = clf.predict_proba(vec.transform(X))
    classes = np.array(clf.classes_, dtype=object)
    best_idx = np.argmax(proba, axis=1)
    forced = classes[best_idx]
    conf = proba[np.arange(len(X)), best_idx]

    sweep = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = forced.copy()
        pred[conf < t] = OTHER
        pack = _metric_pack(y, pred, risk_score=conf)
        sweep.append({
            "threshold": round(float(t), 2),
            "accuracy_8way": pack["accuracy_8way"],
            "macro_f1_present_labels": pack["macro_f1_present_labels"],
            "macro_f1_full_taxonomy": pack["macro_f1_full_taxonomy"],
            "risk_f1": pack["risk_detection"]["f1"],
            "risk_precision": pack["risk_detection"]["precision"],
            "risk_recall": pack["risk_detection"]["recall"],
        })
    best_by_risk_f1 = max(sweep, key=lambda r: r["risk_f1"])
    best_by_macro = max(sweep, key=lambda r: r["macro_f1_present_labels"])
    return {
        "forced_7class_no_other": _add_ci(_metric_pack(y, forced, risk_score=np.ones(len(y))), y, forced),
        "confidence_threshold_oracle": {
            "note": "Threshold selected on the evaluation labels; this is an optimistic diagnostic upper bound, not a deployable estimate.",
            "best_by_risk_f1": best_by_risk_f1,
            "best_by_macro_f1_present_labels": best_by_macro,
            "sweep": sweep,
        },
    }


def _new_lr():
    return LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")


def cv_flat_8class(X, y):
    n_splits = min(5, min(Counter(y).values()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.empty(len(y), dtype=object)
    risk_score = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        vec = _vectorizer()
        Xtr = vec.fit_transform(X[tr])
        Xte = vec.transform(X[te])
        clf = _new_lr().fit(Xtr, y[tr])
        oof[te] = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        risk_cols = [i for i, c in enumerate(clf.classes_) if c != OTHER]
        risk_score[te] = proba[:, risk_cols].sum(axis=1)
    pack = _add_ci(_metric_pack(y, oof, risk_score=risk_score), y, oof)
    pack["n_splits"] = n_splits
    return pack, oof, risk_score


def cv_two_stage(X, y):
    n_splits = min(5, min(Counter(y).values()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.empty(len(y), dtype=object)
    risk_score = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        vec = _vectorizer()
        Xtr = vec.fit_transform(X[tr])
        Xte = vec.transform(X[te])

        y_bin = _risk_binary(y)
        det = _new_lr().fit(Xtr, y_bin[tr])
        bin_pred = det.predict(Xte)
        risk_col = list(det.classes_).index(1)
        risk_score[te] = det.predict_proba(Xte)[:, risk_col]

        risk_tr = tr[y[tr] != OTHER]
        typer = _new_lr().fit(Xtr[y[tr] != OTHER], y[risk_tr])
        typed = typer.predict(Xte)
        pred = typed.astype(object)
        pred[bin_pred == 0] = OTHER
        oof[te] = pred
    pack = _add_ci(_metric_pack(y, oof, risk_score=risk_score), y, oof)
    pack["n_splits"] = n_splits
    return pack, oof, risk_score


def binomial_majority_test(y_true, y_pred):
    """One-sided test: model accuracy greater than always-OTHER majority."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    majority_acc = float(np.mean(y_true == OTHER))
    correct = int(np.sum(y_true == y_pred))
    return {
        "majority_acc": round(majority_acc, 4),
        "model_acc": round(float(correct / len(y_true)), 4),
        "alternative": "model accuracy > always-OTHER accuracy",
        "p_value": round(float(stats.binomtest(correct, len(y_true), majority_acc, alternative="greater").pvalue), 5),
    }


def save_oof(rows, flat_pred, flat_score, two_pred, two_score):
    path = os.path.join(TABLE_DIR, "open_set_oof_predictions.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["i", "lang", "gold", "title", "flat_pred", "flat_risk_score", "two_stage_pred", "two_stage_risk_score"])
        for r, fp, fs, tp, ts in zip(rows, flat_pred, flat_score, two_pred, two_score):
            w.writerow([r["i"], r["lang"], r["gold"], r["title"], fp, round(float(fs), 4), tp, round(float(ts), 4)])
    return path


def main():
    rows, no_consensus = build_open_gold()
    X, y, langs = _texts_labels(rows)
    label_dist = Counter(y.tolist())
    risk_n = sum(v for k, v in label_dist.items() if k != OTHER)
    other_n = label_dist[OTHER]

    base = baselines(y)
    synth = synthetic_forced_and_threshold(X, y)
    flat, flat_pred, flat_score = cv_flat_8class(X, y)
    two, two_pred, two_score = cv_two_stage(X, y)
    pred_path = save_oof(rows, flat_pred, flat_score, two_pred, two_score)

    out = {
        "hypotheses": {
            "H1": "A 7-class model without OTHER is not a deployment-like triage model on neutral news streams.",
            "H2": "Explicit open-set learning improves relevance rejection, but type classification remains modest under imbalance.",
            "H3": "Synthetic confidence thresholding is an optimistic diagnostic, not a substitute for real open-set labels.",
        },
        "data": {
            "batch_n": len(rows) + len(no_consensus),
            "consensus_n": len(rows),
            "no_consensus_n": len(no_consensus),
            "risk_consensus_n": int(risk_n),
            "other_consensus_n": int(other_n),
            "risk_prevalence": round(float(risk_n / len(rows)), 4),
            "label_dist": dict(label_dist),
            "lang_dist": dict(Counter(langs.tolist())),
            "open_gold_path": os.path.join(DATA_DIR, "open_gold.jsonl"),
            "oof_predictions_path": pred_path,
        },
        "baselines": base,
        "synthetic_transfer_open_stream": synth,
        "real_open_set_cv": {
            "flat_8class_lr": flat,
            "two_stage_detector_then_typer": two,
            "accuracy_vs_always_other": {
                "flat_8class_lr": binomial_majority_test(y, flat_pred),
                "two_stage_detector_then_typer": binomial_majority_test(y, two_pred),
            },
        },
        "interpretation": {
            "primary_result": (
                "Open-set evaluation changes the claim: the system is not a deployable natural-stream monitor unless it "
                "models OTHER/relevance. Real open-set training can reject many non-risk items, but risk typing remains "
                "limited and class-imbalanced."
            ),
            "reporting_rule": (
                "Report open-set risk-detection F1/AUPRC, 8-way macro-F1, and risk-only macro-F1 together. Do not headline "
                "closed-set accuracy alone."
            ),
        },
    }
    _write_json(os.path.join(RESULTS_DIR, "open_set_eval.json"), out)
    return out


if __name__ == "__main__":
    res = main()
    print("=== Open-set SCRM triage evaluation ===")
    print(json.dumps({
        "data": res["data"],
        "always_other": {
            "accuracy": res["baselines"]["always_other"]["accuracy_8way"],
            "risk_f1": res["baselines"]["always_other"]["risk_detection"]["f1"],
        },
        "synthetic_forced": {
            "accuracy": res["synthetic_transfer_open_stream"]["forced_7class_no_other"]["accuracy_8way"],
                "macro_f1_present": res["synthetic_transfer_open_stream"]["forced_7class_no_other"]["macro_f1_present_labels"],
                "macro_f1_full_taxonomy": res["synthetic_transfer_open_stream"]["forced_7class_no_other"]["macro_f1_full_taxonomy"],
                "risk_f1": res["synthetic_transfer_open_stream"]["forced_7class_no_other"]["risk_detection"]["f1"],
            },
        "flat_8class_cv": {
            "accuracy": res["real_open_set_cv"]["flat_8class_lr"]["accuracy_8way"],
            "macro_f1_present": res["real_open_set_cv"]["flat_8class_lr"]["macro_f1_present_labels"],
            "macro_f1_full_taxonomy": res["real_open_set_cv"]["flat_8class_lr"]["macro_f1_full_taxonomy"],
            "risk_f1": res["real_open_set_cv"]["flat_8class_lr"]["risk_detection"]["f1"],
            "risk_type_macro_f1_present": res["real_open_set_cv"]["flat_8class_lr"]["risk_type_macro_f1_present_risk_labels"],
        },
        "two_stage_cv": {
            "accuracy": res["real_open_set_cv"]["two_stage_detector_then_typer"]["accuracy_8way"],
            "macro_f1_present": res["real_open_set_cv"]["two_stage_detector_then_typer"]["macro_f1_present_labels"],
            "macro_f1_full_taxonomy": res["real_open_set_cv"]["two_stage_detector_then_typer"]["macro_f1_full_taxonomy"],
            "risk_f1": res["real_open_set_cv"]["two_stage_detector_then_typer"]["risk_detection"]["f1"],
            "risk_type_macro_f1_present": res["real_open_set_cv"]["two_stage_detector_then_typer"]["risk_type_macro_f1_present_risk_labels"],
        },
    }, ensure_ascii=False, indent=2))
