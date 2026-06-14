"""
preprocess.py — 이중언어 전처리 (리포트 METHODOLOGY③).

한국어(교착어): KoNLPy Okt 형태소 분석 → 명사·동사·형용사 어간만 추출.
영어: NLTK 토큰화 + WordNet 표제어(lemma) + 불용어 제거.
두 언어를 동일 인터페이스 tokenize(text, lang) 로 통합 → 단일 코퍼스 학습.
"""
from __future__ import annotations
import json, re
from functools import lru_cache

# ---- 영어 ----
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

_EN_STOP = set(stopwords.words("english")) | {
    "say", "said", "says", "would", "could", "also", "amid", "may", "across",
    "reuters", "ap", "according", "report", "reports", "news",
}
_lemmatizer = WordNetLemmatizer()

def _wn_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def tokenize_en(text: str):
    toks = word_tokenize(text.lower())
    toks = [t for t in toks if t.isalpha() and len(t) >= 2]
    tagged = pos_tag(toks)
    out = []
    for w, tag in tagged:
        lemma = _lemmatizer.lemmatize(w, _wn_pos(tag))
        if lemma in _EN_STOP or len(lemma) < 2:
            continue
        out.append(lemma)
    return out

# ---- 한국어 ----
from konlpy.tag import Okt
_okt = Okt()

# 도메인/일반 불용어(명사로 분류되나 의미 약한 것) + Okt 오분할 보정
_KO_STOP = {
    "것", "수", "등", "및", "관련", "업계", "분석", "전망", "여파", "이로", "인해",
    "대한", "통해", "위해", "가운데", "한편", "지난", "올해", "내년", "오늘", "기자",
    "이번", "당국", "정부", "사태", "영향", "일대", "전망이다", "분석된다",
}
# Okt 가 자주 틀리는 도메인어 보정 사전 (오분할 토큰 → 정정 토큰)
_KO_FIX = {
    "초로": "좌초", "좌": None, "반도": "반도체", "체": None,
}

def tokenize_ko(text: str):
    # 대괄호 태그 등 제거
    text = re.sub(r"\[[^\]]*\]", " ", text)
    pairs = _okt.pos(text, norm=True, stem=True)
    out = []
    for w, p in pairs:
        if p not in ("Noun", "Verb", "Adjective"):
            continue
        w = _KO_FIX.get(w, w)
        if w is None:
            continue
        if w in _KO_STOP or len(w) < 2:
            continue
        out.append(w)
    return out

def tokenize(text: str, lang: str):
    return tokenize_ko(text) if lang == "ko" else tokenize_en(text)

# ---- 코퍼스 I/O ----
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def tokenize_docs(docs, text_key="text"):
    """docs(list[dict]) 각각에 'tokens' 추가하여 반환."""
    for d in docs:
        d["tokens"] = tokenize(d.get(text_key, ""), d.get("lang", "en"))
    return docs


if __name__ == "__main__":
    from config import CORPUS_PATH
    docs = load_jsonl(CORPUS_PATH)
    print(f"코퍼스 {len(docs)}건 로드")
    print("\n[한국어 전처리 예시]")
    ko = [d for d in docs if d["lang"] == "ko"][:3]
    for d in ko:
        print(f"  원문: {d['text']}")
        print(f"  토큰: {tokenize_ko(d['text'])}\n")
    print("[영어 전처리 예시]")
    en = [d for d in docs if d["lang"] == "en"][:3]
    for d in en:
        print(f"  원문: {d['text']}")
        print(f"  토큰: {tokenize_en(d['text'])}\n")
