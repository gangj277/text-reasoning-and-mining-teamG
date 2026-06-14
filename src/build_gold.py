"""
build_gold.py — 사람-검수 골드셋 확정 (외부 타당성 검증의 핵심).

gold_candidates.json(실제 헤드라인 112건)을 연구자가 직접 읽고 7유형 정답을 부여.
다중원인·분류체계 밖(사이버/사기)·원인불명 건은 drop 하여 골드 품질을 확보.
query_hint(약한 힌트)와 무관하게 '주 리스크 메커니즘' 기준으로 단일 라벨 부여.
결과: data/gdelt_gold.jsonl  (필드: title, lang, gold)
"""
import json
from config import DATA_DIR, GOLD_PATH, RISK_CODES
import os

# 연구자 수기 라벨 (index → 정답코드). dict 에 없는 index = drop.
LABELS = {
    # GEOPOLITICAL
    0: "GEOPOLITICAL", 1: "GEOPOLITICAL", 2: "GEOPOLITICAL", 3: "GEOPOLITICAL",
    4: "GEOPOLITICAL", 5: "GEOPOLITICAL", 6: "GEOPOLITICAL", 7: "GEOPOLITICAL",
    8: "GEOPOLITICAL", 9: "GEOPOLITICAL", 10: "GEOPOLITICAL", 11: "GEOPOLITICAL",
    12: "GEOPOLITICAL", 13: "GEOPOLITICAL", 14: "GEOPOLITICAL", 15: "GEOPOLITICAL",
    89: "GEOPOLITICAL",  # 쿠웨이트 항만 피격 = 지정학 충돌
    # LOGISTICS
    16: "LOGISTICS", 17: "LOGISTICS", 18: "LOGISTICS", 19: "LOGISTICS",
    20: "LOGISTICS", 21: "LOGISTICS", 22: "LOGISTICS", 23: "LOGISTICS",
    24: "LOGISTICS", 25: "LOGISTICS", 26: "LOGISTICS", 27: "LOGISTICS",
    28: "LOGISTICS", 29: "LOGISTICS", 30: "LOGISTICS", 31: "LOGISTICS",
    # NATURAL_DISASTER  (drop 33,34,46)
    32: "NATURAL_DISASTER", 35: "NATURAL_DISASTER", 36: "NATURAL_DISASTER",
    37: "NATURAL_DISASTER", 38: "NATURAL_DISASTER", 39: "NATURAL_DISASTER",
    40: "NATURAL_DISASTER", 41: "NATURAL_DISASTER", 42: "NATURAL_DISASTER",
    43: "NATURAL_DISASTER", 44: "NATURAL_DISASTER", 45: "NATURAL_DISASTER",
    47: "NATURAL_DISASTER",
    # SUPPLIER  (drop 62 = 사이버공격)
    48: "SUPPLIER", 49: "SUPPLIER", 50: "SUPPLIER", 51: "SUPPLIER", 52: "SUPPLIER",
    53: "SUPPLIER", 54: "SUPPLIER", 55: "SUPPLIER", 56: "SUPPLIER", 57: "SUPPLIER",
    58: "SUPPLIER", 59: "SUPPLIER", 60: "SUPPLIER", 61: "SUPPLIER", 63: "SUPPLIER",
    # FINANCIAL  (drop 64 일반론, 70 피싱사기)
    65: "FINANCIAL", 66: "FINANCIAL", 67: "FINANCIAL", 68: "FINANCIAL", 69: "FINANCIAL",
    71: "FINANCIAL", 72: "FINANCIAL", 73: "FINANCIAL", 74: "FINANCIAL", 75: "FINANCIAL",
    76: "FINANCIAL", 77: "FINANCIAL", 78: "FINANCIAL", 79: "FINANCIAL",
    # LABOR  (drop 82 정책일반론, 87 다중원인; 89→GEO 위에서)
    80: "LABOR", 81: "LABOR", 83: "LABOR", 84: "LABOR", 85: "LABOR", 86: "LABOR",
    88: "LABOR", 90: "LABOR", 91: "LABOR", 92: "LABOR", 93: "LABOR", 94: "LABOR",
    95: "LABOR",
    # PANDEMIC  (drop 97 원인불명, 101 원인불명 폐쇄)
    96: "PANDEMIC", 98: "PANDEMIC", 99: "PANDEMIC", 100: "PANDEMIC", 102: "PANDEMIC",
    103: "PANDEMIC", 104: "PANDEMIC", 105: "PANDEMIC", 106: "PANDEMIC", 107: "PANDEMIC",
    108: "PANDEMIC", 109: "PANDEMIC", 110: "PANDEMIC", 111: "PANDEMIC",
}


def build():
    cands = json.load(open(os.path.join(DATA_DIR, "gold_candidates.json"), encoding="utf-8"))
    by_i = {c["i"]: c for c in cands}
    gold = []
    for i, code in LABELS.items():
        assert code in RISK_CODES, code
        c = by_i[i]
        gold.append({"title": c["title"], "lang": c["lang"], "gold": code, "src_hint": c["hint"]})
    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return gold


if __name__ == "__main__":
    from collections import Counter
    g = build()
    print(f"골드셋 {len(g)}건 확정 (후보 112 중 {112-len(g)}건 drop)  저장: {GOLD_PATH}")
    print("유형분포:", dict(Counter(x["gold"] for x in g)))
    print("언어분포:", dict(Counter(x["lang"] for x in g)))
    # 힌트와 최종라벨이 다른(연구자가 교정한) 건수
    diff = sum(1 for x in g if x["gold"] != x["src_hint"])
    print(f"질의힌트→정답 교정 건수: {diff}")
