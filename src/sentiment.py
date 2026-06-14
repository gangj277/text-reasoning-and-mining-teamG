"""
sentiment.py — 감성 기반 심각도 점수화 (Pang & Lee 2008).

부정 극성의 '강도'를 심각도(LOW/MED/HIGH)로 환산한다.
 - 영어: NLTK VADER + 도메인 강도 사전 보강.
 - 한국어: 도메인 부정-강도 사전(severe=3 / moderate=2 / mild=1).
검증: 합성 코퍼스의 정답 심각도(생성시 부여)에 대해 점수가 단조 상승하는지
      (Spearman ρ, 3분위 빈닝 정확도, 레벨별 평균점수).
결과: results/rq_severity.json
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import spearmanr
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config import RESULTS_DIR, SEVERITY_ORD, GOLD_PATH
from preprocess import load_jsonl, tokenize
from config import CORPUS_PATH

_sia = SentimentIntensityAnalyzer()

# 도메인 부정-강도 사전 (가중치 3=심각 / 2=중간 / 1=경미)
SEV_LEX = {
    "ko": {
        3: ["마비", "붕괴", "대란", "직격탄", "셧다운", "폐쇄", "사상", "막대", "쇼크", "비상",
             "위기", "전면", "차단", "급락", "충격"],
        2: ["급등", "폭등", "중단", "차질", "타격", "손실", "부족", "혼잡", "적체", "파업",
             "단축", "악화", "심화", "감산"],
        1: ["지연", "둔화", "압박", "부담", "감소"],
    },
    "en": {
        3: ["paralyze", "paralyzed", "collapse", "crisis", "shutdown", "halt", "severe",
             "massive", "cripple", "crippled", "catastrophic", "devastate", "shock", "cut"],
        2: ["surge", "spike", "disruption", "disrupt", "shortage", "strike", "plunge", "slump",
             "soar", "hit", "damage", "stall", "snarl"],
        1: ["delay", "slow", "pressure", "decline"],
    },
}
# 헤지(약화) 표현 — 심각도를 낮춤 (감성강도 원리: 'slightly/소폭'은 강도 감쇄)
HEDGE = {
    "ko": ["우려", "소폭", "일부", "경미", "단기", "다소", "가능성", "제기"],
    "en": ["minor", "slight", "limited", "modest", "may", "anticipate", "soft", "ease", "possible"],
}


def severity_score(text, lang):
    """심각도 점수 (높을수록 심각). 강도어 가산 − 헤지 감산, 길이 보정."""
    toks = tokenize(text, lang)
    if not toks:
        return 0.0
    w = 0.0
    hedges = set(HEDGE[lang])
    for t in toks:
        if any(t.startswith(h) or t == h for h in hedges):
            w -= 1.0                        # 헤지: 강도 감쇄
            continue
        for weight, words in SEV_LEX[lang].items():
            if any(t.startswith(x) or t == x for x in words):
                w += weight
                break
    lex = w / (len(toks) ** 0.5)            # 길이 보정
    if lang == "en":                        # 영어는 VADER 부정도를 보강
        neg = -_sia.polarity_scores(text)["compound"]  # 부정일수록 +
        lex += max(0.0, neg) * 1.5
    return lex


def to_level(scores):
    """점수를 3분위로 LOW/MED/HIGH 빈닝."""
    q1, q2 = np.quantile(scores, [1/3, 2/3])
    out = []
    for s in scores:
        out.append("LOW" if s <= q1 else ("MEDIUM" if s <= q2 else "HIGH"))
    return out


def run():
    docs = load_jsonl(CORPUS_PATH)
    scores = np.array([severity_score(d["text"], d["lang"]) for d in docs])
    true = [d["severity"] for d in docs]
    true_ord = np.array([SEVERITY_ORD[s] for s in true])

    rho, p = spearmanr(scores, true_ord)
    pred_level = to_level(scores)
    acc = float(np.mean([pred_level[i] == true[i] for i in range(len(docs))]))
    # 인접 허용(±1단계) 정확도
    pred_ord = np.array([SEVERITY_ORD[s] for s in pred_level])
    adj = float(np.mean(np.abs(pred_ord - true_ord) <= 1))
    mean_by_level = {lv: round(float(scores[[i for i in range(len(docs)) if true[i] == lv]].mean()), 3)
                     for lv in ("LOW", "MEDIUM", "HIGH")}

    per_lang = {}
    for lg in ("ko", "en"):
        idx = [i for i in range(len(docs)) if docs[i]["lang"] == lg]
        r, _ = spearmanr(scores[idx], true_ord[idx])
        per_lang[lg] = round(float(r), 4)

    out = {"spearman_rho": round(float(rho), 4), "p_value": float(p),
           "tertile_acc": round(acc, 4), "adjacent_acc": round(adj, 4),
           "mean_score_by_true_level": mean_by_level, "per_lang_rho": per_lang}
    with open(os.path.join(RESULTS_DIR, "rq_severity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 실데이터 골드에 심각도 적용(정성 데모, end-to-end 출력 예시)
    gold = load_jsonl(GOLD_PATH)
    gscores = [severity_score(g["title"], g["lang"]) for g in gold]
    glevel = to_level(np.array(gscores))
    demo = [{"title": gold[i]["title"], "lang": gold[i]["lang"], "risk": gold[i]["gold"],
             "severity": glevel[i]} for i in range(len(gold))]
    with open(os.path.join(RESULTS_DIR, "severity_demo.json"), "w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, indent=2)
    return out, demo


if __name__ == "__main__":
    print("=== 감성 기반 심각도 검증 ===")
    out, demo = run()
    print(f"  Spearman ρ = {out['spearman_rho']}  (p={out['p_value']:.2e})")
    print(f"  3분위 정확도 = {out['tertile_acc']} | ±1단계 정확도 = {out['adjacent_acc']}")
    print(f"  레벨별 평균점수(단조 상승해야): {out['mean_score_by_true_level']}")
    print(f"  언어별 ρ: {out['per_lang_rho']}")
    print("\n[실데이터 심각도 출력 예시]")
    for d in demo[:5]:
        from config import CODE2KO
        print(f"   [{d['lang']}] {CODE2KO[d['risk']]} · 심각도 {d['severity']} | {d['title'][:50]}")
