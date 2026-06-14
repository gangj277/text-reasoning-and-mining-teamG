"""
realnews.py — 실제 뉴스 헤드라인 수집 (Google News RSS, 키 불필요, 한·영).

GDELT DOC API가 429 throttle 로 실패하여 Google News RSS 로 교체.
유형별·언어별 질의로 실제 공급망 리스크 헤드라인을 수집한다.
query_type 은 '약한 힌트'(어떤 질의로 잡혔는지)일 뿐, 골드 라벨이 아니다.
골드 라벨은 build_gold.py 에서 사람-검수로 확정한다.
결과: data/realnews_raw.jsonl
"""
from __future__ import annotations
import json, os, re, time, ssl, socket, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from config import DATA_DIR

REALNEWS_RAW = os.path.join(DATA_DIR, "realnews_raw.jsonl")

_ctx = ssl.create_default_context(); _ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
socket.setdefaulttimeout(20)

QUERIES = {
    "GEOPOLITICAL": {"ko": ["수출통제 공급망", "관세 무역분쟁 공급망"],
                     "en": ["export controls supply chain", "tariffs trade war supply chain"]},
    "LOGISTICS": {"ko": ["항만 적체 물류 대란", "해운 운임 공급망 지연"],
                  "en": ["port congestion shipping delay", "freight container supply chain"]},
    "NATURAL_DISASTER": {"ko": ["지진 공장 공급망", "태풍 홍수 공급망 피해"],
                         "en": ["earthquake factory supply chain", "flood supply chain disruption"]},
    "SUPPLIER": {"ko": ["반도체 공급 부족", "공장 가동중단 부품 공급"],
                 "en": ["chip shortage supplier", "factory shutdown production halt"]},
    "FINANCIAL": {"ko": ["원자재 가격 급등 공급망", "환율 원자재 비용 상승"],
                  "en": ["commodity prices supply chain", "raw material cost inflation supply"]},
    "LABOR": {"ko": ["파업 물류 공급망", "노조 파업 공장 생산"],
              "en": ["port strike supply chain", "labor strike factory production"]},
    "PANDEMIC": {"ko": ["감염병 봉쇄 공급망", "방역 공장 폐쇄 생산"],
                 "en": ["lockdown supply chain factory", "outbreak factory closure supply"]},
}


def _rss(q, lang):
    if lang == "ko":
        base = "https://news.google.com/rss/search?q={}&hl=ko&gl=KR&ceid=KR:ko"
    else:
        base = "https://news.google.com/rss/search?q={}&hl=en-US&gl=US&ceid=US:en"
    url = base.format(urllib.parse.quote(q))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, context=_ctx).read()
    root = ET.fromstring(raw)
    out = []
    for it in root.findall(".//item"):
        t = it.find("title")
        if t is not None and t.text:
            out.append(t.text.strip())
    return out


def _clean(title):
    # Google News 는 ' - 매체명' 을 끝에 붙임 → 마지막 ' - ' 이후 제거
    return re.sub(r"\s+-\s+[^-]+$", "", title).strip()


def fetch(sleep=1.5):
    rows = []
    for code, bylang in QUERIES.items():
        for lang, qs in bylang.items():
            for q in qs:
                try:
                    titles = _rss(q, lang)
                except Exception as e:
                    print(f"  [{code}/{lang}] '{q}' FAIL {e}"); titles = []
                n0 = len(rows)
                for t in titles:
                    ct = _clean(t)
                    if len(ct) >= 12:
                        rows.append({"title": ct, "lang": lang, "query_type": code})
                print(f"  [{code}/{lang}] '{q[:30]}' -> +{len(rows)-n0}")
                time.sleep(sleep)
    # dedup
    seen, uniq = set(), []
    for r in rows:
        k = r["title"].lower()
        if k not in seen:
            seen.add(k); uniq.append(r)
    with open(REALNEWS_RAW, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return uniq


if __name__ == "__main__":
    from collections import Counter
    print("=== Google News RSS 실데이터 수집 ===")
    rows = fetch()
    print(f"\n총 {len(rows)}건 (dedup)  저장: {REALNEWS_RAW}")
    print("언어:", Counter(r["lang"] for r in rows))
    print("질의유형:", Counter(r["query_type"] for r in rows))
