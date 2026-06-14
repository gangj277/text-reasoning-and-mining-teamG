"""
severity_gold.py — 심각도 순환성 해소 (적대적 검토 critical #1 대응).

문제: 합성 심각도 검증(ρ=0.672)은 정답을 정의한 단어를 lexicon이 그대로 채점한 순환 구조.
해소: 연구자가 lexicon 과 무관하게 '실제 뉴스 헤드라인'의 심각도를 독립 평정한 골드를 만들고,
      lexicon 점수가 이 독립 라벨과 상관되는지 검증한다. (실제 disruption 규모 기준 평정:
      가동중단/붕괴/전쟁/항만피격=HIGH · 차질/지연/가격상승/파업진행=MEDIUM · 우려/전망/전략/해소=LOW)
결과: results/severity_independent.json
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import spearmanr
from config import RESULTS_DIR, GOLD_PATH, SEVERITY_ORD
from preprocess import load_jsonl
from sentiment import severity_score, to_level

# 연구자 독립 심각도 평정 (gold 파일 행 순서 0..101, lexicon 비참조). 블록별 7유형 순.
RATINGS = (
  # GEOPOLITICAL 0-16 (17)
  ["LOW","MEDIUM","MEDIUM","LOW","MEDIUM","MEDIUM","LOW","MEDIUM","LOW","MEDIUM","LOW",
   "MEDIUM","MEDIUM","LOW","MEDIUM","LOW","HIGH"] +
  # LOGISTICS 17-32 (16)
  ["LOW","HIGH","MEDIUM","LOW","HIGH","LOW","HIGH","MEDIUM","MEDIUM","MEDIUM","LOW",
   "MEDIUM","LOW","LOW","LOW","MEDIUM"] +
  # NATURAL_DISASTER 33-45 (13)
  ["HIGH","LOW","HIGH","MEDIUM","MEDIUM","MEDIUM","HIGH","LOW","LOW","MEDIUM","MEDIUM",
   "MEDIUM","MEDIUM"] +
  # SUPPLIER 46-60 (15)
  ["MEDIUM","HIGH","HIGH","LOW","HIGH","HIGH","HIGH","MEDIUM","MEDIUM","MEDIUM","HIGH",
   "MEDIUM","MEDIUM","MEDIUM","LOW"] +
  # FINANCIAL 61-74 (14)
  ["MEDIUM","LOW","MEDIUM","MEDIUM","MEDIUM","MEDIUM","MEDIUM","MEDIUM","MEDIUM","LOW",
   "MEDIUM","LOW","LOW","MEDIUM"] +
  # LABOR 75-87 (13)
  ["MEDIUM","MEDIUM","MEDIUM","MEDIUM","MEDIUM","MEDIUM","MEDIUM","MEDIUM","MEDIUM",
   "MEDIUM","MEDIUM","LOW","LOW"] +
  # PANDEMIC 88-101 (14)
  ["LOW","LOW","MEDIUM","LOW","HIGH","MEDIUM","LOW","HIGH","MEDIUM","MEDIUM","LOW",
   "HIGH","MEDIUM","HIGH"]
)


def main():
    gold = load_jsonl(GOLD_PATH)
    assert len(gold) == len(RATINGS), (len(gold), len(RATINGS))
    scores = np.array([severity_score(g["title"], g["lang"]) for g in gold])
    true_ord = np.array([SEVERITY_ORD[r] for r in RATINGS])

    rho, p = spearmanr(scores, true_ord)
    pred_level = to_level(scores)
    acc = float(np.mean([pred_level[i] == RATINGS[i] for i in range(len(gold))]))
    pred_ord = np.array([SEVERITY_ORD[s] for s in pred_level])
    adj = float(np.mean(np.abs(pred_ord - true_ord) <= 1))
    mean_by = {lv: round(float(scores[[i for i in range(len(gold)) if RATINGS[i] == lv]].mean()), 3)
               for lv in ("LOW", "MEDIUM", "HIGH")}
    from collections import Counter
    out = {"n": len(gold), "independent_spearman_rho": round(float(rho), 4), "p_value": float(p),
           "tertile_acc": round(acc, 4), "adjacent_acc": round(adj, 4),
           "mean_score_by_independent_level": mean_by,
           "rating_distribution": dict(Counter(RATINGS)),
           "note": "lexicon 과 독립적으로 평정한 실데이터 심각도 라벨 대비 검증 (순환성 해소)"}
    json.dump(out, open(os.path.join(RESULTS_DIR, "severity_independent.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    o = main()
    print("=== 독립 심각도 검증 (순환성 해소) ===")
    print(f"  n={o['n']}  평정분포={o['rating_distribution']}")
    print(f"  독립 Spearman ρ = {o['independent_spearman_rho']}  (p={o['p_value']:.2e})")
    print(f"  3분위 정확도 = {o['tertile_acc']} | ±1단계 = {o['adjacent_acc']}")
    print(f"  독립레벨별 평균점수(단조 기대): {o['mean_score_by_independent_level']}")
