"""
topic_lda.py — RQ1: 비지도 토픽모델(LDA)이 공급망 리스크 유형을 자동 발견하는가?

[방법론적 발견]
 BoW LDA를 한·영 통합 코퍼스에 그대로 적용하면 토픽이 '리스크 유형'이 아니라
 '언어'로 먼저 분리된다(두 언어가 어휘를 공유하지 않기 때문). 이는 BoW 토픽모델의
 알려진 한계다. 따라서 본 연구는 (1) naive 통합 LDA를 baseline 으로 보고하고,
 (2) 언어 stratified LDA(언어별 학습)를 주 분석으로 채택한다.

[추가 처리]
 7유형이 공유하는 글루 어휘(문법동사 + 공통 결과어)는 토픽 변별력이 없으므로
 토픽 전용 불용어로 제거 → 변별적 '원인(event)' 어휘가 토픽을 정의하게 한다.

절차: 언어별 K(4~10) 스윕 → c_v 최적 K → 토픽 상위어 → 정답라벨 사상 purity/복원유형수.
결과: results/rq1_lda.json, results/tables/lda_topic_terms.csv
"""
from __future__ import annotations
import json, os, csv
from collections import Counter, defaultdict
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from config import SEED, RESULTS_DIR, TABLE_DIR, CODE2KO, CORPUS_PATH
from preprocess import load_jsonl, tokenize_docs

# 토픽 전용 불용어 — 유형 변별력이 없는 공유 글루(문법동사 + 공통 결과/규모어)
TOPIC_STOP = {
    "ko": {"하다", "되다", "돼다", "있다", "이르다", "지다", "인하다", "생기다", "나타나다",
           "빚어지다", "제기", "가능성", "분석", "전망", "영향", "관련", "업계", "여파", "일대",
           "가운데", "한편", "차질", "지연", "우려", "소폭", "일부", "경미", "단기", "부담",
           "상승", "부족", "사실", "막대", "손실", "조달", "재고", "공급", "기업", "거론"},
    "en": {"follow", "event", "impact", "reach", "near", "come", "result", "across", "several",
           "day", "sector", "may", "full", "halt", "minor", "slight", "limited", "modest",
           "supply", "disruption", "delay", "delivery", "cost", "rise", "spread", "emerge",
           "appear", "shortage", "inventory", "procurement", "schedule", "slip", "postpone",
           "cut", "paralyze", "operation", "firm", "loss", "massive", "severe", "friction",
           "anticipate", "expect", "likely", "report", "week", "month", "record"},
}


def build_bow(token_lists, lang):
    filt = [[w for w in toks if w not in TOPIC_STOP[lang]] for toks in token_lists]
    dictionary = Dictionary(filt)
    dictionary.filter_extremes(no_below=3, no_above=0.4)
    bow = [dictionary.doc2bow(t) for t in filt]
    return dictionary, bow, filt


def train_lda(bow, dictionary, k):
    return LdaModel(corpus=bow, id2word=dictionary, num_topics=k, random_state=SEED,
                    passes=15, iterations=150, alpha="auto", eta="auto", eval_every=None)


def coherence_cv(model, texts, dictionary):
    return float(CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                                coherence="c_v").get_coherence())


def map_topics(model, bow, labels, k):
    topic_label = defaultdict(Counter)
    for b, lab in zip(bow, labels):
        if not b:
            continue
        dom = max(model.get_document_topics(b, minimum_probability=0.0), key=lambda x: x[1])[0]
        topic_label[dom][lab] += 1
    rows = []
    for t in range(k):
        c = topic_label.get(t, Counter()); total = sum(c.values())
        if total == 0:
            rows.append({"topic": t, "mapped_risk": "—", "purity": 0.0, "size": 0}); continue
        risk, cnt = c.most_common(1)[0]
        rows.append({"topic": t, "mapped_risk": risk, "purity": round(cnt/total, 3), "size": total})
    recovered = len({r["mapped_risk"] for r in rows if r["mapped_risk"] != "—"})
    weighted_purity = sum(r["purity"]*r["size"] for r in rows) / max(1, sum(r["size"] for r in rows))
    return rows, recovered, round(weighted_purity, 3)


def run_lda_one(docs, lang, k_min=4, k_max=10):
    token_lists = [d["tokens"] for d in docs]
    labels = [d["risk_type"] for d in docs]
    dictionary, bow, texts = build_bow(token_lists, lang)
    coh, models = {}, {}
    for k in range(k_min, k_max+1):
        m = train_lda(bow, dictionary, k)
        coh[k] = coherence_cv(m, texts, dictionary); models[k] = m
        print(f"    [{lang}] K={k:2d}  c_v={coh[k]:.4f}")
    best_k = max(coh, key=coh.get); best = models[best_k]
    mapping, recovered, wpur = map_topics(best, bow, labels, best_k)
    topics = []
    for t in range(best_k):
        terms = [w for w, _ in best.show_topic(t, topn=10)]
        info = next(m for m in mapping if m["topic"] == t)
        topics.append({"topic": t, "terms": terms, "mapped_risk": info["mapped_risk"],
                       "purity": info["purity"], "size": info["size"]})
    print(f"    [{lang}] → best K={best_k} c_v={coh[best_k]:.4f} | 복원 {recovered}/7 | 가중purity {wpur}")
    return {"lang": lang, "coherence_by_k": {str(k): round(v, 4) for k, v in coh.items()},
            "best_k": best_k, "best_cv": round(coh[best_k], 4),
            "recovered_types": recovered, "weighted_purity": wpur, "topics": topics}


def run_rq1():
    docs = tokenize_docs(load_jsonl(CORPUS_PATH))

    # (1) naive 통합 LDA baseline — 언어로 분리되는지 확인용
    print("  [naive 통합 LDA baseline]")
    all_tokens = [d["tokens"] for d in docs]
    # 통합은 불용어를 양 언어 합집합으로 적용
    merged_stop = TOPIC_STOP["ko"] | TOPIC_STOP["en"]
    filt = [[w for w in t if w not in merged_stop] for t in all_tokens]
    dct = Dictionary(filt); dct.filter_extremes(no_below=3, no_above=0.4)
    bow = [dct.doc2bow(t) for t in filt]
    m6 = train_lda(bow, dct, 6)
    naive_cv = coherence_cv(m6, filt, dct)
    # 토픽이 언어로 갈리는 정도: 각 토픽 지배 문서의 언어 순도
    langs = [d["lang"] for d in docs]
    tl = defaultdict(Counter)
    for b, lg in zip(bow, langs):
        if b:
            tl[max(m6.get_document_topics(b, minimum_probability=0.0), key=lambda x: x[1])[0]][lg] += 1
    lang_purity = sum(max(c.values()) for c in tl.values()) / max(1, sum(sum(c.values()) for c in tl.values()))
    print(f"    naive K=6 c_v={naive_cv:.4f} | 토픽의 언어순도={lang_purity:.3f}"
          f"  (→ {lang_purity*100:.0f}%가 단일언어로 분리)")

    # (2) 언어 stratified LDA — 주 분석
    print("  [언어 stratified LDA]")
    ko = run_lda_one([d for d in docs if d["lang"] == "ko"], "ko")
    en = run_lda_one([d for d in docs if d["lang"] == "en"], "en")

    out = {"naive_combined": {"k": 6, "cv": round(naive_cv, 4), "lang_purity": round(lang_purity, 3)},
           "stratified": {"ko": ko, "en": en},
           "headline_cv": round((ko["best_cv"] + en["best_cv"]) / 2, 4),
           "headline_recovered": f'ko {ko["recovered_types"]}/7, en {en["recovered_types"]}/7'}
    with open(os.path.join(RESULTS_DIR, "rq1_lda.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(os.path.join(TABLE_DIR, "lda_topic_terms.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["lang", "topic", "mapped_risk(ko)", "purity", "size", "top_terms"])
        for sec in (ko, en):
            for tp in sec["topics"]:
                w.writerow([sec["lang"], tp["topic"], CODE2KO.get(tp["mapped_risk"], tp["mapped_risk"]),
                            tp["purity"], tp["size"], " ".join(tp["terms"])])
    return out


if __name__ == "__main__":
    print("=== RQ1: LDA 토픽 발견 ===")
    res = run_rq1()
    print(f"\n  headline c_v(언어평균)={res['headline_cv']} | 복원 {res['headline_recovered']}")
    print("\n[언어별 발견 토픽]")
    for lg in ("ko", "en"):
        print(f"  <{lg}>  best K={res['stratified'][lg]['best_k']}  c_v={res['stratified'][lg]['best_cv']}")
        for tp in res["stratified"][lg]["topics"]:
            print(f"    T{tp['topic']} → {CODE2KO.get(tp['mapped_risk'], tp['mapped_risk'])}"
                  f" (purity {tp['purity']}, n={tp['size']}): {' '.join(tp['terms'][:7])}")
