"""
research_gate.py — fixed high-standard KPI gate for research framing.

The gate is intentionally multi-metric. Accuracy alone is not acceptable because
OTHER dominates the open-set stream. A defensible research claim must show:

  - useful open-set triage,
  - balanced risk/non-risk behavior,
  - meaningful type classification beyond relevance detection,
  - reduced root-cause error buckets.
"""
from __future__ import annotations

import json
import os

from config import RESULTS_DIR

DEFAULT_GATE = {
    "accuracy_8way": 0.85,
    "macro_f1_present_labels": 0.65,
    "risk_precision": 0.84,
    "risk_recall": 0.82,
    "risk_f1": 0.85,
    "risk_auprc": 0.90,
    "risk_type_macro_f1_present": 0.65,
    "max_false_positive_risk": 30,
    "max_false_negative_risk": 40,
    "max_risk_type_wrong": 25,
}


def _get_metric(metrics, key):
    if key == "risk_precision":
        return metrics["risk_detection"]["precision"]
    if key == "risk_recall":
        return metrics["risk_detection"]["recall"]
    if key == "risk_f1":
        return metrics["risk_detection"]["f1"]
    if key == "risk_auprc":
        return metrics["risk_detection"]["average_precision"]
    if key == "risk_type_macro_f1_present":
        return metrics["risk_type_macro_f1_present_risk_labels"]
    return metrics[key]


def evaluate_gate(metrics, error_counts=None, gate=None):
    gate = gate or DEFAULT_GATE
    failed = {}
    observed = {}
    for key in (
        "accuracy_8way",
        "macro_f1_present_labels",
        "risk_precision",
        "risk_recall",
        "risk_f1",
        "risk_auprc",
        "risk_type_macro_f1_present",
    ):
        value = _get_metric(metrics, key)
        observed[key] = value
        if value < gate[key]:
            failed[key] = {"observed": value, "required": gate[key]}

    if error_counts:
        error_checks = {
            "false_positive_risk": "max_false_positive_risk",
            "false_negative_risk": "max_false_negative_risk",
            "risk_type_wrong": "max_risk_type_wrong",
        }
        for error_key, gate_key in error_checks.items():
            value = error_counts.get(error_key, 0)
            observed[error_key] = value
            if value > gate[gate_key]:
                failed[error_key] = {"observed": value, "maximum": gate[gate_key]}

    return {
        "passed": not failed,
        "failed": failed,
        "observed": observed,
        "gate": gate,
    }


def evaluate_mechanism_results():
    path = os.path.join(RESULTS_DIR, "mechanism_engine_eval.json")
    data = json.load(open(path, encoding="utf-8"))
    metrics = data["mechanism_two_stage_cv"]
    errors = data["error_breakdown"]["mechanism_two_stage"]["counts"]
    result = evaluate_gate(metrics, errors)
    out_path = os.path.join(RESULTS_DIR, "research_gate.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate_mechanism_results(), ensure_ascii=False, indent=2))
