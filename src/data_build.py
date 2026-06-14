"""
data_build.py — 데이터 계층.

(A) 사건 기반 구조화 이중언어 코퍼스 생성 (백본, 정답 라벨 + 정답 심각도)
(B) GDELT DOC 2.0 API로 실제 뉴스 헤드라인 수집 (외적 타당성 검증용, best-effort)

설계 원칙:
 - 한국어/영어를 '번역'이 아니라 독립 저작 → 언어별 성능 비교(RQ3)가 공정.
 - 유형별 '원인(event)' 어휘는 변별적이되, '결과(consequence)' 어휘는 7유형이 공유
   → 분류가 키워드 1:1 매칭으로 풀리지 않게(=trivial 분리 방지).
 - 심각도(severity)는 결과 어휘 강도 + 규모(scale)로 부여 → 감성기반 심각도의 정답.
 - 모든 무작위는 SEED 고정 → 100% 재현.
"""
from __future__ import annotations
import json, random, time, ssl, socket, urllib.parse, urllib.request
from config import (RISK_TYPES, SEED, CORPUS_PATH, GDELT_RAW_PATH, SEVERITY_LEVELS)

# ============================================================================
# (A) 구조화 코퍼스
# ============================================================================

# 7유형이 공유하는 결과(consequence) 어휘 — 심각도 3단계
CONSEQ = {
    "ko": {
        "LOW":    ["소폭 차질이 우려된다", "일부 지연 가능성이 제기됐다", "경미한 영향이 예상된다",
                   "단기 비용 부담이 거론된다", "공급에 다소 차질이 빚어질 전망이다"],
        "MEDIUM": ["공급 차질이 빚어졌다", "납기 지연이 확산되고 있다", "생산 일정에 차질이 생겼다",
                   "조달 비용이 상승했다", "재고 부족이 나타나고 있다"],
        "HIGH":   ["공급망이 사실상 마비됐다", "전면 중단 사태로 번졌다", "심각한 공급 대란이 빚어졌다",
                   "납품이 전면 차단됐다", "기업들이 막대한 손실을 입었다"],
    },
    "en": {
        "LOW":    ["minor delays are expected", "a slight cost increase is anticipated",
                   "limited disruption is likely", "some shipments may be postponed",
                   "modest supply friction is reported"],
        "MEDIUM": ["supply disruptions have emerged", "delivery delays are spreading",
                   "production schedules have slipped", "procurement costs have risen",
                   "inventory shortages are appearing"],
        "HIGH":   ["the supply chain has been paralyzed", "operations came to a full halt",
                   "a severe supply crisis has erupted", "deliveries were completely cut off",
                   "firms suffered massive losses"],
    },
}

# 규모(scale) 표현 — 심각도별 강도
SCALE = {
    "ko": {"LOW": ["{n}%", "수일간"], "MEDIUM": ["{n}%", "수주간", "{n}억 원"],
           "HIGH": ["{n}%", "수개월간", "{n}조 원", "사상 최대"]},
    "en": {"LOW": ["{n}%", "for several days"], "MEDIUM": ["{n}%", "for weeks", "${n}M"],
           "HIGH": ["{n}%", "for months", "${n}B", "a record"]},
}

# 유형별 원인(event)·주체(actor)·장소(location) 뱅크
BANK = {
    "GEOPOLITICAL": {
        "ko": {"actor": ["미국 정부", "중국 당국", "EU 집행위", "러시아", "수출 당국", "양국 정상"],
               "event": ["대중 반도체 수출통제를 강화", "고율 관세를 전격 부과", "핵심 광물 수출을 제한",
                         "경제 제재를 발표", "무역 분쟁이 격화", "전략물자 통관을 봉쇄"],
               "loc": ["워싱턴", "베이징", "브뤼셀", "동유럽", "대만해협", "중동"]},
        "en": {"actor": ["the US government", "Chinese authorities", "the EU", "Russia", "trade regulators"],
               "event": ["tightened chip export controls", "imposed steep tariffs", "restricted critical-mineral exports",
                         "announced fresh sanctions", "escalated a trade dispute", "blocked strategic-goods clearance"],
               "loc": ["Washington", "Beijing", "Brussels", "Eastern Europe", "the Taiwan Strait"]},
    },
    "LOGISTICS": {
        "ko": {"actor": ["해운업계", "항만 당국", "물류 대기업", "선사", "운송업체"],
               "event": ["수에즈 운하에서 선박이 좌초", "항만에 컨테이너가 적체", "운임이 폭등", "해운 노선이 마비",
                         "주요 항로가 봉쇄", "내륙 운송망이 두절"],
               "loc": ["수에즈 운하", "부산항", "로테르담항", "상하이항", "파나마 운하", "LA항"]},
        "en": {"actor": ["the shipping industry", "port authorities", "a major logistics firm", "carriers"],
               "event": ["a vessel ran aground in the Suez Canal", "containers piled up at the port",
                         "freight rates spiked", "shipping lanes were paralyzed", "a key route was blockaded",
                         "inland transport was severed"],
               "loc": ["the Suez Canal", "the Port of Los Angeles", "Rotterdam", "Shanghai", "the Panama Canal"]},
    },
    "NATURAL_DISASTER": {
        "ko": {"actor": ["기상 당국", "재난 본부", "현지 정부", "구조 당국"],
               "event": ["규모 7의 강진이 발생", "기록적 홍수가 산업단지를 강타", "초강력 태풍이 상륙",
                         "장기 가뭄이 지속", "대형 산불이 확산", "폭우로 공장 지대가 침수"],
               "loc": ["대만 신주", "일본 규슈", "동남아", "미 텍사스", "중국 쓰촨", "유럽 라인강"]},
        "en": {"actor": ["weather agencies", "disaster officials", "local authorities"],
               "event": ["a magnitude-7 earthquake struck", "record floods hit an industrial cluster",
                         "a super typhoon made landfall", "a prolonged drought persisted",
                         "a massive wildfire spread", "torrential rain submerged factory zones"],
               "loc": ["Hsinchu, Taiwan", "Kyushu, Japan", "Southeast Asia", "Texas", "Sichuan", "the Rhine"]},
    },
    "SUPPLIER": {
        "ko": {"actor": ["반도체 공급사", "1차 협력사", "부품 제조사", "완성차 공장", "소재 업체"],
               "event": ["핵심 부품 공장이 화재로 가동을 중단", "반도체 생산라인이 멈춤", "협력사 납품이 끊김",
                         "주력 공장이 셧다운", "차량용 칩 생산이 차질", "소재 공급이 끊김"],
               "loc": ["기흥 공장", "히로시마 공장", "텍사스 팹", "선전 공장", "울산 공장"]},
        "en": {"actor": ["a chip supplier", "a tier-1 supplier", "a parts maker", "a carmaker's plant"],
               "event": ["a key component plant halted after a fire", "a semiconductor line went down",
                         "supplier deliveries were cut", "a flagship plant was shut down",
                         "automotive-chip output stalled", "material supply was interrupted"],
               "loc": ["the Giheung fab", "a Hiroshima plant", "a Texas fab", "a Shenzhen plant"]},
    },
    "FINANCIAL": {
        "ko": {"actor": ["원자재 시장", "외환 시장", "중앙은행", "상품 거래소", "에너지 시장"],
               "event": ["국제 유가가 급등", "구리·니켈 가격이 폭등", "환율이 급변동", "원자재 인플레이션이 심화",
                         "기준금리가 인상", "천연가스 가격이 치솟음"],
               "loc": ["뉴욕상품거래소", "런던금속거래소", "국제 유가", "외환시장", "원자재 시장"]},
        "en": {"actor": ["commodity markets", "the FX market", "the central bank", "energy markets"],
               "event": ["crude oil prices surged", "copper and nickel prices spiked", "the currency swung sharply",
                         "commodity inflation intensified", "the benchmark rate was hiked", "gas prices soared"],
               "loc": ["the NYMEX", "the London Metal Exchange", "global oil markets", "the FX market"]},
    },
    "LABOR": {
        "ko": {"actor": ["항만 노조", "트럭 운전자", "완성차 노조", "물류 노동자", "공장 근로자"],
               "event": ["전면 파업에 돌입", "무기한 총파업을 예고", "태업에 들어감", "임금 협상이 결렬",
                         "구인난이 심화", "노사 갈등이 격화"],
               "loc": ["미 서부 항만", "독일 완성차 공장", "부산항", "영국 물류센터", "프랑스 정유소"]},
        "en": {"actor": ["the dockworkers' union", "truck drivers", "an autoworkers' union", "logistics workers"],
               "event": ["launched a full strike", "threatened an indefinite walkout", "began a slowdown",
                         "saw wage talks collapse", "faced a worsening labor shortage", "escalated a labor dispute"],
               "loc": ["US West Coast ports", "German auto plants", "the Port of Busan", "UK warehouses"]},
    },
    "PANDEMIC": {
        "ko": {"actor": ["보건 당국", "현지 정부", "방역 본부", "WHO"],
               "event": ["신종 감염병이 확산", "대규모 봉쇄 조치가 시행", "확진자 급증으로 공장이 폐쇄",
                         "항만이 방역 격리에 들어감", "집단 감염으로 가동이 중단", "방역 통제가 강화"],
               "loc": ["중국 선전", "베트남 호치민", "인도 첸나이", "동남아 공단", "주요 항만 도시"]},
        "en": {"actor": ["health authorities", "local governments", "quarantine officials", "the WHO"],
               "event": ["a novel outbreak spread", "sweeping lockdowns were imposed",
                         "a case surge shut a factory", "a port entered quarantine",
                         "a cluster infection halted operations", "quarantine controls tightened"],
               "loc": ["Shenzhen", "Ho Chi Minh City", "Chennai", "Southeast Asian clusters", "major port cities"]},
    },
}

TEMPLATES = {
    "ko": [
        "[{loc}] {actor}가 {event}하면서 {conseq}.",
        "{loc}에서 {actor}가 {event}했다. 이로 인해 {conseq}.",
        "{actor}가 {event}하자, 관련 업계에서 {conseq}.",
        "{event} 여파로 {loc} 일대에서 {conseq}. 영향은 {scale}에 이를 것으로 분석된다.",
    ],
    "en": [
        "[{loc}] {actor} {event}, and {conseq}.",
        "In {loc}, {actor} {event}. As a result, {conseq}.",
        "After {actor} {event}, {conseq} across the sector.",
        "Following the event near {loc}, {conseq}. The impact may reach {scale}.",
    ],
}


def _fill_scale(rng, lang, sev):
    tmpl = rng.choice(SCALE[lang][sev])
    return tmpl.format(n=rng.choice([3, 8, 12, 20, 35, 50, 80]))


def _make_doc(rng, code, lang, idx):
    sev = rng.choices(SEVERITY_LEVELS, weights=[0.3, 0.4, 0.3])[0]
    b = BANK[code][lang]
    actor = rng.choice(b["actor"]); event = rng.choice(b["event"]); loc = rng.choice(b["loc"])
    conseq = rng.choice(CONSEQ[lang][sev])
    scale = _fill_scale(rng, lang, sev)
    tmpl = rng.choice(TEMPLATES[lang])
    text = tmpl.format(actor=actor, event=event, loc=loc, conseq=conseq, scale=scale)
    # 헤드라인 = 주체+사건 핵심
    title = f"{actor} {event}" if lang == "ko" else f"{actor} {event}"
    return {"id": f"{code}_{lang}_{idx:04d}", "lang": lang, "risk_type": code,
            "severity": sev, "title": title.strip(), "text": text, "source": "constructed"}


def build_corpus(per_type_per_lang=110):
    rng = random.Random(SEED)
    docs = []
    for r in RISK_TYPES:
        for lang in ("ko", "en"):
            for i in range(per_type_per_lang):
                docs.append(_make_doc(rng, r["code"], lang, i))
    rng.shuffle(docs)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return docs


# ============================================================================
# (B) GDELT 실데이터 (best-effort, 외적 타당성 검증용)
# ============================================================================
_GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
# 유형별 실데이터 질의(영어 + 한국어). query_type 은 '약한 힌트'일 뿐, 골드라벨 아님.
GDELT_QUERIES = {
    "GEOPOLITICAL": ['("export controls" OR sanctions OR tariffs) "supply chain"',
                     '수출통제 OR 제재 OR 관세 공급망 sourcelang:korean'],
    "LOGISTICS":    ['("port congestion" OR "shipping delays" OR "suez canal") supply',
                     '항만 OR 물류대란 OR 해운 운임 sourcelang:korean'],
    "NATURAL_DISASTER": ['(earthquake OR flood OR typhoon) "supply chain" factory',
                     '지진 OR 홍수 OR 태풍 공급망 공장 sourcelang:korean'],
    "SUPPLIER":     ['("chip shortage" OR "factory shutdown" OR "production halt") supplier',
                     '반도체 부족 OR 공장 가동중단 OR 부품 공급 sourcelang:korean'],
    "FINANCIAL":    ['("commodity prices" OR "raw material" inflation) supply chain',
                     '원자재 가격 급등 OR 환율 공급망 sourcelang:korean'],
    "LABOR":        ['(strike OR "labor dispute" OR walkout) port OR logistics OR factory',
                     '파업 항만 OR 물류 OR 공장 sourcelang:korean'],
    "PANDEMIC":     ['(lockdown OR outbreak OR quarantine) "supply chain" factory OR port',
                     '봉쇄 OR 감염 확산 공장 OR 항만 공급망 sourcelang:korean'],
}


def _http_json(url, retries=4):
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    socket.setdefaulttimeout(25)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            return json.loads(urllib.request.urlopen(req, context=ctx).read())
        except Exception as e:
            wait = 3 * (attempt + 1)
            if "429" in str(e) or "Remote end" in str(e) or "timed out" in str(e).lower():
                time.sleep(wait); continue
            time.sleep(wait)
    return None


def fetch_gdelt(maxrecords=60, sleep=4.0):
    """유형별 실제 헤드라인 수집. 네트워크 실패시 빈 리스트(파이프라인은 코퍼스로 계속)."""
    rows = []
    for code, queries in GDELT_QUERIES.items():
        for q in queries:
            params = {"query": q, "mode": "artlist", "maxrecords": str(maxrecords),
                      "format": "json", "sort": "datedesc"}
            url = _GDELT + "?" + urllib.parse.urlencode(params)
            data = _http_json(url)
            n_before = len(rows)
            if data and isinstance(data, dict):
                for a in data.get("articles", []):
                    title = (a.get("title") or "").strip()
                    if len(title) < 12:
                        continue
                    rows.append({"title": title, "url": a.get("url", ""),
                                 "domain": a.get("domain", ""), "language": a.get("language", ""),
                                 "seendate": a.get("seendate", ""), "query_type": code})
            print(f"  [{code}] q='{q[:38]}...' -> +{len(rows)-n_before} (총 {len(rows)})")
            time.sleep(sleep)  # 429 회피
    # 중복 제거(title 기준)
    seen, uniq = set(), []
    for r in rows:
        k = r["title"].lower()
        if k in seen:
            continue
        seen.add(k); uniq.append(r)
    with open(GDELT_RAW_PATH, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return uniq


if __name__ == "__main__":
    print("=== (A) 구조화 이중언어 코퍼스 생성 ===")
    docs = build_corpus()
    from collections import Counter
    print(f"  총 {len(docs)}건  | 언어 {Counter(d['lang'] for d in docs)}")
    print(f"  유형 {Counter(d['risk_type'] for d in docs)}")
    print(f"  심각도 {Counter(d['severity'] for d in docs)}")
    print(f"  저장: {CORPUS_PATH}")
    print("\n  [샘플]")
    for d in docs[:4]:
        print(f"   ({d['lang']}/{d['risk_type']}/{d['severity']}) {d['text']}")

    print("\n=== (B) GDELT 실데이터 수집 (best-effort) ===")
    try:
        g = fetch_gdelt()
        from collections import Counter as C
        print(f"  실데이터 {len(g)}건 저장: {GDELT_RAW_PATH}")
        print(f"  언어분포 {C(r['language'] for r in g)}")
    except Exception as e:
        print("  GDELT 수집 건너뜀:", e)
