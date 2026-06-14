"""
config.py — 전역 설정 / 7대 공급망 리스크 분류체계 / 경로.

본 프레임워크의 모든 모듈이 공유하는 단일 진실 공급원(single source of truth).
리스크 분류체계는 SCRM 문헌(Chu et al. 2020; Nguyen et al. 2025)과
아이디어 리포트의 예시(물류/공급자/금융/지정학/자연재해)를 7대 유형으로 확장한 것.
"""
from __future__ import annotations
import os

# ----------------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
REPORT_DIR = os.path.join(BASE_DIR, "report")
for _d in (DATA_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)

CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")            # 구조화 이중언어 코퍼스
GDELT_RAW_PATH = os.path.join(DATA_DIR, "gdelt_raw.jsonl")      # 실제 GDELT 헤드라인(원본)
GOLD_PATH = os.path.join(DATA_DIR, "gdelt_gold.jsonl")          # 손-라벨링 골드셋
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

SEED = 42  # 전 실험 재현성을 위한 고정 시드

# ----------------------------------------------------------------------------
# 7대 공급망 리스크 유형
#   code        : 내부 식별자(라벨)
#   ko / en     : 표시명
#   seeds_ko/en : Word2Vec 의미공간 확장의 '시드 키워드'(리포트 RQ2 임베딩 확장)
# ----------------------------------------------------------------------------
RISK_TYPES = [
    {
        "code": "GEOPOLITICAL", "ko": "지정학", "en": "Geopolitical",
        "seeds_ko": ["전쟁", "제재", "관세", "수출통제", "분쟁", "갈등", "봉쇄"],
        "seeds_en": ["war", "sanction", "tariff", "export", "conflict", "embargo"],
    },
    {
        "code": "LOGISTICS", "ko": "물류·운송", "en": "Logistics",
        "seeds_ko": ["물류", "운하", "항만", "해운", "컨테이너", "운임", "지연", "선박"],
        "seeds_en": ["logistics", "canal", "port", "shipping", "container", "freight", "delay"],
    },
    {
        "code": "NATURAL_DISASTER", "ko": "자연재해", "en": "Natural Disaster",
        "seeds_ko": ["지진", "홍수", "태풍", "가뭄", "산불", "폭우", "재해"],
        "seeds_en": ["earthquake", "flood", "typhoon", "drought", "wildfire", "storm"],
    },
    {
        "code": "SUPPLIER", "ko": "공급자·생산", "en": "Supplier",
        "seeds_ko": ["공장", "부품", "반도체", "생산", "중단", "가동", "납품"],
        "seeds_en": ["factory", "parts", "semiconductor", "production", "shutdown", "plant"],
    },
    {
        "code": "FINANCIAL", "ko": "금융·원자재", "en": "Financial",
        "seeds_ko": ["원자재", "가격", "급등", "환율", "인플레이션", "금리", "비용"],
        "seeds_en": ["commodity", "price", "surge", "currency", "inflation", "cost"],
    },
    {
        "code": "LABOR", "ko": "노동·파업", "en": "Labor",
        "seeds_ko": ["파업", "노조", "노동", "인력", "임금", "태업", "구인난"],
        "seeds_en": ["strike", "union", "labor", "workforce", "wage", "walkout"],
    },
    {
        "code": "PANDEMIC", "ko": "보건·감염병", "en": "Pandemic",
        "seeds_ko": ["감염병", "봉쇄", "확진", "격리", "방역", "전염", "팬데믹"],
        "seeds_en": ["pandemic", "lockdown", "outbreak", "quarantine", "infection", "virus"],
    },
]

RISK_CODES = [r["code"] for r in RISK_TYPES]
CODE2KO = {r["code"]: r["ko"] for r in RISK_TYPES}
CODE2EN = {r["code"]: r["en"] for r in RISK_TYPES}

# 심각도 3단계 (리포트: 심각도 LOW/MEDIUM/HIGH)
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH"]
SEVERITY_ORD = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
