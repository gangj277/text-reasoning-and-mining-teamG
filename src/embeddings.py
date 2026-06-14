"""
embeddings.py — RQ2 구성요소: Word2Vec 의미공간 기반 시드 키워드 확장 (Mikolov 2013).

 - 언어별 Word2Vec 학습(교착어 한국어/영어는 어휘 비공유 → 언어별 분리 학습).
 - 각 리스크 유형의 '시드 키워드'를 의미공간 최근접이웃으로 확장 → 유형 사전 자동 구축.
 - ablation: 확장 사전 매칭만으로 분류(비지도 lexicon baseline)했을 때 외부 골드 성능을
   측정 → 지도학습(TF-IDF+LR, 외부 F1 0.77)이 더하는 가치를 정량화.
결과: results/tables/word2vec_expansion.csv, results/rq2_embedding.json
"""
from __future__ import annotations
import json, os, csv
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import f1_score, accuracy_score

from config import SEED, RISK_TYPES, RESULTS_DIR, TABLE_DIR, RISK_CODES, CODE2KO, GOLD_PATH
from preprocess import load_jsonl, tokenize_docs, tokenize
from config import CORPUS_PATH


def train_w2v(token_lists):
    return Word2Vec(sentences=token_lists, vector_size=100, window=5, min_count=2,
                    workers=1, sg=1, epochs=60, seed=SEED)


def expand_seeds(model, seeds, topn=6):
    expanded = {}
    for s in seeds:
        if s in model.wv:
            neigh = [w for w, _ in model.wv.most_similar(s, topn=topn)]
            expanded[s] = neigh
    return expanded


def build_lexicons(docs):
    """언어별 Word2Vec 학습 + 유형별 확장 사전."""
    models, lexicons, expansion_rows = {}, {}, []
    for lang in ("ko", "en"):
        toks = [d["tokens"] for d in docs if d["lang"] == lang]
        m = train_w2v(toks); models[lang] = m
        lex = defaultdict(set)
        for r in RISK_TYPES:
            seeds = r["seeds_ko"] if lang == "ko" else r["seeds_en"]
            exp = expand_seeds(m, seeds)
            for s, neigh in exp.items():
                lex[r["code"]].add(s)
                lex[r["code"]].update(neigh)
                expansion_rows.append({"lang": lang, "risk": r["code"], "seed": s,
                                       "expanded": " ".join(neigh)})
        lexicons[lang] = {k: set(v) for k, v in lex.items()}
    return models, lexicons, expansion_rows


def lexicon_classify(tokens, lang, lexicons):
    scores = {c: 0 for c in RISK_CODES}
    lex = lexicons[lang]
    for t in tokens:
        for c in RISK_CODES:
            if t in lex.get(c, ()):
                scores[c] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def run():
    docs = tokenize_docs(load_jsonl(CORPUS_PATH))
    models, lexicons, expansion_rows = build_lexicons(docs)

    # 확장 사전 표 저장
    with open(os.path.join(TABLE_DIR, "word2vec_expansion.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["lang", "risk(ko)", "seed", "expanded_neighbors"])
        for r in expansion_rows:
            w.writerow([r["lang"], CODE2KO[r["risk"]], r["seed"], r["expanded"]])

    # lexicon baseline 을 외부 골드셋에 적용
    gold = load_jsonl(GOLD_PATH)
    yg, pred, covered = [], [], 0
    for g in gold:
        toks = tokenize(g["title"], g["lang"])
        p = lexicon_classify(toks, g["lang"], lexicons)
        yg.append(g["gold"])
        if p is None:
            pred.append("__none__")
        else:
            pred.append(p); covered += 1
    # 커버된 것만으로 F1 (사전 미매칭은 분류불가)
    idx = [i for i in range(len(yg)) if pred[i] != "__none__"]
    macro = float(f1_score([yg[i] for i in idx], [pred[i] for i in idx],
                           average="macro")) if idx else 0.0
    acc_all = float(accuracy_score(yg, pred))  # 미매칭=오답

    out = {"lexicon_external_macro_f1_covered": round(macro, 4),
           "lexicon_external_acc_all": round(acc_all, 4),
           "lexicon_coverage": round(covered / len(gold), 3),
           "supervised_external_acc": 0.7745,  # 비교 기준(외부평가 결과)
           "note": "Word2Vec 확장사전 매칭(비지도) vs TF-IDF+LR(지도) 비교용 ablation"}
    with open(os.path.join(RESULTS_DIR, "rq2_embedding.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out, expansion_rows, lexicons


if __name__ == "__main__":
    print("=== RQ2: Word2Vec 키워드 확장 + lexicon baseline ===")
    out, rows, lex = run()
    print("\n[확장 예시]")
    shown = 0
    for r in rows:
        if r["expanded"] and shown < 10:
            print(f"  ({r['lang']}/{CODE2KO[r['risk']]}) {r['seed']} → {r['expanded']}")
            shown += 1
    print(f"\nlexicon baseline 외부 골드: 커버리지 {out['lexicon_coverage']}, "
          f"커버분 Macro-F1 {out['lexicon_external_macro_f1_covered']}, 전체 정확도 {out['lexicon_external_acc_all']}")
    print(f"→ 지도학습(TF-IDF+LR) 외부 정확도 {out['supervised_external_acc']} 와 비교")
