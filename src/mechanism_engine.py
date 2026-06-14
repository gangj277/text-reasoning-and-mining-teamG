"""
mechanism_engine.py — mechanism-aware bilingual SCRM triage engine.

The baseline engine is a sparse TF-IDF classifier. Its open-set errors show two
structural failures: (1) Korean/English named mechanism terms are often OOV or
fragmented by tokenization, and (2) generic supply-chain strategy/news/cyber
items look lexically similar to real risk events. This module adds a small,
auditable feature layer:

  - word TF-IDF over normalized tokens,
  - character n-grams over raw titles for bilingual/OOV robustness,
  - taxonomy-level mechanism counters plus OTHER/generic/cyber cues.

No generated labels or evaluation examples are used as features. The mechanism
lexicon is derived from the declared taxonomy seeds and general SCRM event
mechanisms.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config import RISK_CODES, RISK_TYPES, SEED
from preprocess import tokenize

OTHER = "OTHER"
OPEN_LABELS = RISK_CODES + [OTHER]

_SEEDS_BY_CODE = {r["code"]: [*r["seeds_ko"], *r["seeds_en"]] for r in RISK_TYPES}

MECHANISM_LEXICON = {
    "GEOPOLITICAL": _SEEDS_BY_CODE["GEOPOLITICAL"] + [
        "희토류", "핵심광물", "중동", "호르무즈", "해협", "원유",
        "rare earth", "critical mineral", "strait", "hormuz", "middle east", "oil supply",
    ],
    "LOGISTICS": _SEEDS_BY_CODE["LOGISTICS"] + [
        "운송", "운항", "항로", "우회", "freight", "route", "transport", "tanker",
        "port performance",
    ],
    "NATURAL_DISASTER": _SEEDS_BY_CODE["NATURAL_DISASTER"] + [
        "화재", "수급 비상", "water risk", "warehouse fire",
    ],
    "SUPPLIER": _SEEDS_BY_CODE["SUPPLIER"] + [
        "수급", "차질", "부족", "대체 공급망", "공급망 제약",
        "component", "memory shortage", "shortage impact", "supply shortage",
    ],
    "FINANCIAL": _SEEDS_BY_CODE["FINANCIAL"] + [
        "유가", "고유가", "물가", "사료 가격", "crude", "oil price",
        "energy crisis", "tanker rates",
    ],
    "LABOR": _SEEDS_BY_CODE["LABOR"] + [
        "총파업", "노사", "협상", "화물연대", "rail strike", "port strike",
    ],
    "PANDEMIC": _SEEDS_BY_CODE["PANDEMIC"],
    "OTHER": [
        "university", "student", "bachelor", "course", "webinar", "forum",
        "roundup", "conference", "award", "competition", "newsletter",
        "ai-driven", "automation", "digital transformation", "strategy", "operations",
        "procurement", "지도", "포럼", "교육", "대학", "학생", "솔루션", "전략",
        "경쟁력", "점검할 시점", "라운드업", "뉴스 모음",
    ],
}

RISK_EVENT_CUES = [
    "crisis", "disruption", "risk", "shock", "stress", "fragility", "shortage", "delay",
    "위기", "리스크", "차질", "비상", "붕괴", "충격", "불안", "흔들", "부담",
]
META_OR_GENERIC_CUES = [
    "news", "roundup", "update", "forum", "conference", "webinar", "report", "announces", "wins",
    "보고서", "포럼", "세미나", "발표", "점검", "기획",
]
CYBER_OTHER_CUES = ["attack", "cyber", "malware", "antivirus", "hacking", "해킹", "사이버", "보안"]


def _count_patterns(text: str, patterns: list[str]) -> int:
    lower = text.lower()
    return sum(1 for pattern in patterns if pattern.lower() in lower)


def mechanism_feature_names() -> list[str]:
    return (
        [f"lex_{code}" for code in OPEN_LABELS]
        + ["cue_risk_event", "cue_meta_or_generic", "cue_cyber_other", "title_length_norm"]
    )


def mechanism_feature_matrix(rows) -> csr_matrix:
    values = []
    for row in rows:
        title = row["title"] if isinstance(row, dict) else str(row)
        feats = []
        for code in OPEN_LABELS:
            feats.append(_count_patterns(title, MECHANISM_LEXICON[code]))
        feats.append(_count_patterns(title, RISK_EVENT_CUES))
        feats.append(_count_patterns(title, META_OR_GENERIC_CUES))
        feats.append(_count_patterns(title, CYBER_OTHER_CUES))
        feats.append(len(title) / 120.0)
        values.append(feats)
    return csr_matrix(np.asarray(values, dtype=float))


def choose_n_splits(labels, requested=5) -> int:
    counts = Counter(labels)
    min_count = min(counts.values())
    if min_count < 2:
        raise ValueError(f"Need at least 2 examples per class for CV; rarest class has {min_count}.")
    return min(requested, min_count)


def token_texts(rows) -> np.ndarray:
    return np.array([" ".join(tokenize(r["title"], r.get("lang", "en"))) for r in rows], dtype=object)


def raw_titles(rows) -> np.ndarray:
    return np.array([r["title"] for r in rows], dtype=object)


def labels(rows) -> np.ndarray:
    return np.array([r["gold"] for r in rows], dtype=object)


def risk_binary(y):
    return (np.asarray(y, dtype=object) != OTHER).astype(int)


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


def metric_pack(y_true, y_pred, risk_score=None):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    present_open = [lab for lab in OPEN_LABELS if np.any(y_true == lab)]
    present_risk = [lab for lab in RISK_CODES if np.any(y_true == lab)]
    y_true_bin = risk_binary(y_true)
    y_pred_bin = risk_binary(y_pred)
    precision, recall, risk_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )
    risk_mask = y_true != OTHER
    return {
        "accuracy_8way": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1_present_labels": round(
            float(f1_score(y_true, y_pred, labels=present_open, average="macro", zero_division=0)), 4
        ),
        "macro_f1_full_taxonomy": round(
            float(f1_score(y_true, y_pred, labels=OPEN_LABELS, average="macro", zero_division=0)), 4
        ),
        "risk_detection": {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(risk_f1), 4),
            "roc_auc": _safe_auc(y_true_bin, risk_score) if risk_score is not None else None,
            "average_precision": _safe_ap(y_true_bin, risk_score) if risk_score is not None else None,
        },
        "risk_type_macro_f1_present_risk_labels": round(
            float(f1_score(y_true[risk_mask], y_pred[risk_mask], labels=present_risk, average="macro", zero_division=0)), 4
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


def _word_vectorizer():
    return TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def _char_vectorizer():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=8000,
        sublinear_tf=True,
    )


def fit_transform_hybrid(train_rows, test_rows):
    train_tokens = token_texts(train_rows)
    test_tokens = token_texts(test_rows)
    train_raw = raw_titles(train_rows)
    test_raw = raw_titles(test_rows)

    word = _word_vectorizer()
    char = _char_vectorizer()
    blocks_train = [
        word.fit_transform(train_tokens),
        char.fit_transform(train_raw),
        mechanism_feature_matrix(train_rows),
    ]
    blocks_test = [
        word.transform(test_tokens),
        char.transform(test_raw),
        mechanism_feature_matrix(test_rows),
    ]
    return hstack(blocks_train).tocsr(), hstack(blocks_test).tocsr()


def new_classifier():
    return LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")


def best_threshold_by_risk_f1(y_true_bin, scores, grid=None):
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for threshold in grid:
        pred = (scores >= threshold).astype(int)
        f1 = f1_score(y_true_bin, pred, zero_division=0)
        # Prefer the lower threshold on exact ties to avoid hiding rare risks.
        if f1 > best_f1 or (f1 == best_f1 and threshold < best_t):
            best_t, best_f1 = float(threshold), float(f1)
    return round(best_t, 4), round(best_f1, 4)


def _binary_threshold_metrics(y_true_bin, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true_bin == 1)))
    fp = int(np.sum((pred == 1) & (y_true_bin == 0)))
    fn = int(np.sum((pred == 0) & (y_true_bin == 1)))
    tn = int(np.sum((pred == 0) & (y_true_bin == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_threshold_for_research_gate(
    y_true_bin,
    scores,
    grid=None,
    min_precision=0.86,
    min_recall=0.82,
):
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    candidates = []
    precision_candidates = []
    recall_candidates = []
    fallback = []
    for threshold in grid:
        metrics = _binary_threshold_metrics(y_true_bin, scores, float(threshold))
        item = (float(threshold), metrics)
        fallback.append(item)
        if metrics["precision"] >= min_precision and metrics["recall"] >= min_recall:
            candidates.append(item)
        if metrics["precision"] >= min_precision:
            precision_candidates.append(item)
        if metrics["recall"] >= min_recall:
            recall_candidates.append(item)

    if candidates:
        threshold, metrics = max(
            candidates,
            key=lambda item: (
                item[1]["f1"],
                item[1]["precision"],
                item[1]["recall"],
                item[0],
            ),
        )
        metrics = dict(metrics)
        metrics["fallback"] = False
    else:
        fallback_source = "precision_floor_only"
        fallback_pool = precision_candidates
        if not fallback_pool:
            fallback_source = "recall_floor_only"
            fallback_pool = recall_candidates
        if not fallback_pool:
            fallback_source = "raw_f1_only"
            fallback_pool = fallback
        threshold, metrics = max(
            fallback_pool,
            key=lambda item: (
                item[1]["f1"],
                item[1]["precision"],
                item[1]["recall"],
                item[0],
            ),
        )
        metrics = dict(metrics)
        metrics["fallback"] = True
        metrics["fallback_reason"] = f"no_threshold_met_both_floors__used_{fallback_source}"

    rounded = {
        key: (round(float(value), 4) if isinstance(value, (float, np.floating)) else value)
        for key, value in metrics.items()
    }
    return round(float(threshold), 4), rounded


def calibrate_detector_threshold(X_train, y_train_bin, requested_splits=3):
    counts = Counter(y_train_bin)
    if len(counts) < 2 or min(counts.values()) < 2:
        return 0.5, {"method": "fallback", "reason": "insufficient binary class support"}
    n_splits = min(requested_splits, min(counts.values()))
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y_train_bin), dtype=float)
    for tr, va in inner.split(np.zeros(len(y_train_bin)), y_train_bin):
        detector = new_classifier().fit(X_train[tr], y_train_bin[tr])
        risk_col = list(detector.classes_).index(1)
        scores[va] = detector.predict_proba(X_train[va])[:, risk_col]
    threshold, threshold_metrics = best_threshold_for_research_gate(y_train_bin, scores)
    return threshold, {
        "method": "inner_cv_gate_aware",
        "n_splits": n_splits,
        "inner_threshold_metrics": threshold_metrics,
    }


def run_mechanism_cv(rows, requested_splits=5, mode="two_stage"):
    y = labels(rows)
    n_splits = choose_n_splits(y, requested=requested_splits)
    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pred = np.empty(len(rows), dtype=object)
    risk_score = np.zeros(len(rows), dtype=float)
    thresholds = []

    rows_arr = np.asarray(rows, dtype=object)
    for train_idx, test_idx in splits.split(np.zeros(len(y)), y):
        train_rows = rows_arr[train_idx].tolist()
        test_rows = rows_arr[test_idx].tolist()
        X_train, X_test = fit_transform_hybrid(train_rows, test_rows)
        y_train = y[train_idx]

        if mode == "flat":
            clf = new_classifier().fit(X_train, y_train)
            pred[test_idx] = clf.predict(X_test)
            prob = clf.predict_proba(X_test)
            risk_cols = [i for i, c in enumerate(clf.classes_) if c != OTHER]
            risk_score[test_idx] = prob[:, risk_cols].sum(axis=1)
        elif mode == "two_stage":
            y_bin = risk_binary(y_train)
            detector = new_classifier().fit(X_train, y_bin)
            risk_col = list(detector.classes_).index(1)
            risk_score[test_idx] = detector.predict_proba(X_test)[:, risk_col]
            threshold, threshold_info = calibrate_detector_threshold(X_train, y_bin)
            threshold_info["threshold"] = threshold
            thresholds.append(threshold_info)
            bin_pred = (risk_score[test_idx] >= threshold).astype(int)

            risk_train_mask = y_train != OTHER
            typer = new_classifier().fit(X_train[risk_train_mask], y_train[risk_train_mask])
            typed_pred = typer.predict(X_test).astype(object)
            typed_pred[bin_pred == 0] = OTHER
            pred[test_idx] = typed_pred
        else:
            raise ValueError(f"Unknown mode: {mode}")

    result = metric_pack(y, pred, risk_score=risk_score)
    result["n_splits"] = n_splits
    if thresholds:
        result["detector_thresholds"] = thresholds
    return result, pred, risk_score
