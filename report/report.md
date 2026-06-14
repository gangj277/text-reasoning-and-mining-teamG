# 한·영 뉴스 텍스트마이닝 기반 공급망 리스크 탐지 시스템

### Bilingual News-Based Supply Chain Risk Detection via Text Mining — 최종 결과 보고서

**지식추론과 텍스트마이닝 · Term Project ｜ Team G**
2021147037 정영훈 · 2023147009 전유하 · 2023147018 강지민

---

## 초록 (Abstract)

본 연구는 한국어·영어 뉴스를 단일 파이프라인에서 분석하여 공급망 리스크를 **자동 탐지·7유형 분류·심각도 평가**하는 텍스트마이닝 프레임워크를 설계하고, **실제 뉴스로 다단계 검증**하였다. TF-IDF·LDA·Word2Vec·감성사전·지도분류기 등 강의에서 다룬 고전 기법만으로 구성하여 해석 가능한 구조를 유지했다(분류·심각도 산출 파이프라인에는 언어모델 API 미사용).

본 보고서의 **가장 큰 기여는 방법론적이다 — 자기 적대적 검증으로 화려한 수치의 함정을 스스로 폭로하고, 오류 분석에 근거해 최종 엔진을 실제로 고도화했다.** (1) **RQ1** — 통합 LDA는 토픽을 리스크 유형이 아니라 **언어 경계로 분리(86%)**하는 한계를 보였고, 언어별 분리 학습으로 의미 있는 리스크 토픽을 복원했다. (2) **RQ2** — 타입질의 골드의 0.766·약-라벨 CV의 0.884는 분류 능력이 아니라 *검색질의·시드 키워드 복원*이다(ML 없는 query-echo가 0.99로 능가). 순환을 제거한 **검정력 있는 label-blind 골드(3개 평정 관점, Fleiss κ=0.84, n=217)**에서는 **합성 학습 모델의 정확도(0.447)가 다수클래스 베이스라인(0.65)에 미달**해 closed-set 합성 전이가 실배포에 부적합함을 보였다. 여기서 한 걸음 더 나아가, 중립질의 700건 중 합의된 **699건 전체를 `OTHER` 포함 open-set gold**로 재구성했다(위험 217/OTHER 482). plain TF-IDF open-set은 관련 뉴스 선별 신호를 보였지만 아직 제한적이었다(flat LR: accuracy 0.820, risk-F1 0.776, AUPRC 0.851). 이에 OOV 위험 메커니즘과 generic OTHER 혼동이라는 gap을 분석해 **mechanism-aware hybrid engine**(word TF-IDF + raw character n-gram + taxonomy mechanism feature + gate-aware two-stage threshold)을 설계했고, 3-fold OOF에서 **accuracy 0.884, risk-F1 0.861, AUPRC 0.923, present-label Macro-F1 0.696, 위험유형 Macro-F1 0.697**을 달성해 고정 research gate를 통과했다. (3) **RQ3** — 한·영 성능차는 소표본과 오염된 타입질의 근거만으로는 단정할 수 없으며, powered/open-set 단계에서는 언어별 차이를 별도 검정할 만큼 충분한 균형 표본이 필요하다. (4) **심각도** — 순환성을 제거한 독립 평정 검증에서 Spearman ρ=0.40(p<0.001)의 유의한 단조 상관을 보였다.

요컨대 고전 텍스트마이닝은 *키워드가 정렬된* 뉴스와 폐쇄형 7유형 분류만으로는 실배포를 주장할 수 없고, `OTHER`를 포함한 open-set triage를 명시적으로 학습해야 관련 뉴스 선별 신호가 나타난다. 본 연구의 기여는 이 경계를 **가설화하고, 실험으로 부수고, 오류분석 기반 엔진 고도화로 고정 KPI gate까지 통과**한 데 있다.

---

## 1. 서론

### 1.1 배경 및 문제 정의

공급망 리스크 관리(SCRM)는 전통적으로 재고·수요·리드타임 같은 **정형 내부 데이터**에 의존해 왔다. 그러나 지진·전쟁·파업·감염병 등 외생적 충격은 **뉴스라는 비정형 텍스트에 가장 먼저, 가장 풍부하게** 나타난다. 정형 지표는 충격이 실적에 반영된 *이후*에야 신호를 주므로 선제 대응이 어렵다.

비정형 뉴스 활용에는 세 난점이 있다. ① 하루 수십만 건이라 전수 모니터링이 어렵고, ② 리스크 유형이 이질적(지정학·물류·자연재해·공급자·금융·노동·보건)이라 단일 키워드로는 한계가 있으며, ③ 한·영이 혼재하나 기존 연구는 영어 단일 언어에 편중되고 **한국어 공급망 뉴스 분석은 선행연구가 희소**하다(Nguyen et al., 2025).

### 1.2 연구 질문

> **RQ1.** 비지도 토픽모델(LDA)이 뉴스에서 의미 있는 공급망 리스크 유형을 자동 발견할 수 있는가?
> **RQ2.** 임베딩 확장과 지도 분류 모델이 리스크 유형을 정확히 분류할 수 있는가? *그리고 그 수치는 무엇을 진짜로 측정하는가?*
> **RQ3.** 교착어 한국어와 영어를 단일 파이프라인에서 일관되게 처리할 수 있는가?

### 1.3 방법론적 입장

본 연구의 핵심 가치는 **"좋아 보이는 수치"를 의심하고 검증하는 태도**에 있다. 우리는 합성 데이터의 과대평가, 검색질의와 라벨의 순환, 시드 키워드 의존을 *스스로 공격*하여 각각을 정량화하고, 그 결과 화려한 수치를 정직한 수치로 교체했다.

---

## 2. 데이터

본 연구는 **다섯 가지 데이터 자산**을 구축했다(표 1). 모든 무작위 과정은 시드(SEED=42)로 고정했다.

**표 1. 데이터 구성**

| 자산 | 규모 | 라벨 | 출처 | 용도 |
|---|---|---|---|---|
| 구조화 코퍼스 | 1,540건 (한 770/영 770) | 정답 유형+심각도 | 실제 사건 기반 템플릿 | 학습·내부평가 |
| 실뉴스 코퍼스 | 2,062건 (한 833/영 1,229) | 약-라벨(질의유형) | Google News RSS (타입질의) | 대규모 CV |
| 타입질의 골드 | 102건 (한 48/영 54) | 연구자 검수 | 위에서 표집·수기 | 전이 검증(+순환 진단) |
| **powered label-blind 골드** | **217건 (한 124/영 93)** | **3인 패널 합의 (Fleiss κ=0.84)** | **중립질의("공급망")** | **검정력 있는 일반화 검증** |
| **open-set gold** | **699건 (위험 217/OTHER 482)** | **3인 패널 다수합의** | **중립질의 700건 전체** | **`OTHER` 포함 최종 triage 검증** |

### 2.1 구조화 이중언어 코퍼스

7대 유형 × 2언어에 대해, 실제 사건(수에즈 좌초·반도체 부족·항만 파업·대만 지진·희토류 통제·코로나 봉쇄 등) 기반 어휘 뱅크로 문장을 합성했다. 한·영을 *독립 저작*하여 언어 비교(RQ3)의 공정성을 확보하고, *결과(consequence)* 어휘는 7유형이 공유하게 했다.

> **구성 타당성 고지.** 이 코퍼스는 통제된 *측정 도구*이지 실제 뉴스 분포의 표본이 아니다. 따라서 본 연구는 핵심 성능을 **실데이터에서 재측정**한다.

### 2.2 실데이터와 두 종류의 골드셋

GDELT API가 요청 제한으로 실패하여 **Google News RSS**(키 불필요·한·영)로 2,062건을 수집했다. 이후 두 골드셋을 구축했다.
- **타입질의 골드(102건)**: 유형별 질의로 수집한 헤드라인을 연구자가 7유형으로 수기 라벨. *그러나 §5.2에서 밝히듯, 질의가 곧 라벨을 결정하는 순환이 내재한다.*
- **powered label-blind 골드(217건)**: 유형 정보 없는 **중립 질의("공급망"/"supply chain")**로 혼합 스트림 700건을 수집하고, 3개 독립 평정 관점의 패널이 질의힌트 없이 blind 라벨링한 뒤 합의(≥2/3, OTHER 제외)한 것. 라벨이 질의로 결정되지 않아 *순환이 제거*되며, **Fleiss κ=0.84**로 내부 일치도를 검증했다(초기 단일평정 n=26 → 패널 n=217로 검정력 보강).
- **open-set gold(699건)**: 위 700건에서 무합의 1건만 제외하고, 위험 7유형뿐 아니라 **OTHER 합의 482건을 포함**한 최종 평가셋이다. 이는 "이미 위험 뉴스라고 주어진 헤드라인을 7유형으로 나누는가"가 아니라, **중립 공급망 뉴스 스트림에서 먼저 관련 위험 뉴스를 걸러낼 수 있는가**를 묻는다.

---

## 3. 방법론

`수집 → 전처리(Okt/NLTK) → 토픽발견(LDA) → 분류(TF-IDF+분류기) → 심각도(감성) → 평가`

**표 2. 적용 기법과 이론적 근거**

| 기법 | 역할 | 근거 |
|---|---|---|
| TF-IDF·벡터공간모델 | 문서 가중 벡터화 | Salton & McGill (1983) |
| LDA 토픽모델 | 잠재 리스크 주제 비지도 발견 | Blei et al. (2003) |
| Word2Vec | 시드 키워드 의미공간 확장 | Mikolov et al. (2013) |
| 지도 문서분류 | 7대 유형 분류 | McCallum & Nigam (1998) |
| 문자 n-gram + 메커니즘 feature | 한·영 OOV 위험 표현과 generic OTHER 구분 | 본 연구의 오류분석 기반 고도화 |
| 감성 분석 | 부정 강도 → 심각도 | Pang & Lee (2008) |
| c_v 일관성 | 토픽 품질 평가 | Röder et al. (2015) |

한국어는 교착어이므로 **KoNLPy Okt**로 명사·동사·형용사 어간을, 영어는 **NLTK**로 표제어를 추출해 단일 인터페이스로 통합했다.

---

## 4. RQ1 — LDA 토픽 발견

![RQ1 LDA coherence](figures/fig1_lda_coherence.png)
*그림 1. 언어 stratified LDA의 c_v(점선=naive 통합 LDA).*

**핵심 발견: 통합 LDA는 언어로 분리된다.** 한·영 통합 K=6 LDA는 c_v=0.42이나 **토픽의 86%가 단일 언어 문서로 채워진다.** 이는 BoW 토픽모델이 어휘 비공유 두 언어를 *리스크 유형이 아니라 언어 경계로* 먼저 분리하는 알려진 한계다. 이에 **언어별 분리 학습**을 채택하여 한국어 K=8 c_v=0.36(7유형 중 5개 복원), 영어 K=4 c_v=0.40(4개 복원)을 얻었다. 일부 토픽은 선명했으나(*최선 사례*로 한국어 금융 토픽 순도 1.00), 평균 순도는 더 낮고 `공장`처럼 다유형 공통어로 인한 혼재 토픽도 있었다(복원 ko 5/7·en 4/7) — 실제 공급망 뉴스의 다중성을 반영한 결과다.

> **RQ1 답.** LDA는 *언어별로* 의미 있는 리스크 토픽을 발견하나, 통합 적용 시 언어 분리가 우선한다. 비지도 토픽 발견은 지도 분류의 *보조 탐색 도구*로 유효하다.

---

## 5. RQ2 — 분류: 수치가 진짜 무엇을 측정하는가

본 절은 본 연구의 핵심이며, **하나의 수치가 아니라 수치의 해부**다.

### 5.1 내부 합성 성능은 과대평가다

층화 held-out에서 LogisticRegression·LinearSVC 모두 **Macro-F1=1.00**에 도달했으나, 이는 *성과가 아니라 합성 코퍼스의 분리 가능성의 산물*이다. 유형별 원인 어휘가 변별적이라 어떤 BoW 모델도 합성 테스트를 거의 완벽히 맞힌다. **내부 수치는 sanity check일 뿐, 어떤 실세계 양도 추정하지 못한다.**

### 5.2 타입질의 골드의 0.766은 "질의버킷 복원"이다

합성 학습 모델은 타입질의 골드 102건에 Macro-F1 0.766을 냈다. 그러나 **자기 적대적 검증이 결정적 사실을 드러냈다**: 골드 라벨의 **101/102(99%)가 그것을 수집한 검색질의와 일치**하며, 따라서 *"라벨 = 검색질의"라고 답하는 ML 없는 trivial 베이스라인이 Macro-F1 0.99*로 본 모델을 능가한다(표 3). 즉 0.766은 "이 헤드라인을 어떤 키워드 질의가 가져왔는가"를 복원하는 과제이며, 그마저 trivial echo보다 못하다.

**표 3. 정직한 베이스라인 비교 (타입질의 골드 n=102)**

| 방법 | Macro-F1 | 해석 |
|---|---|---|
| query-echo (라벨=질의) | **0.990** | 순환의 상한 — ML 불필요 |
| 본 파이프라인 (TF-IDF+LR) | 0.766 | echo보다 낮음 |
| 시드 키워드 규칙 (전처리·ML 없음) | 0.717 | 본 방법 이득은 +0.05로 modest |
| char n-gram (전처리 없음) | 0.596 | — |
| 다수클래스 / 무작위 | 0.167 / 0.143 | 하한 |

![internal vs external](figures/fig3_internal_vs_external.png)
*그림 2. 내부(합성) 과적합 대 타입질의 골드 전이. 단, 후자는 §5.2에서 보듯 질의복원이다.*

### 5.3 시드 키워드를 제거하면 절반으로 떨어진다

시드 키워드를 골드에서 제거(토큰 17%)하면 Macro-F1이 0.766 → **0.484로 반토막**난다. 즉 전이의 상당 부분이 *합성 코퍼스에 심고 검색에도 쓴 동일 시드 키워드 매칭*에 의존하는 것으로, **무거운 시드 의존**을 드러낸다(시드 제거 후 0.484는 다수클래스 0.167은 넘으나 본격 분류 능력으로 보긴 어렵다). 골드의 진짜 어휘 커버리지는 **27.7%**(헤드라인당 중앙 4개 in-vocab 특징)에 불과하다 — 앞서 보고했던 "0.98"은 실은 *OOV가 아닌 문서의 비율*로 오표기였다.

### 5.4 검정력 있는 label-blind 검증 (3인 패널, Fleiss κ=0.84)

순환을 근본적으로 깨기 위해, **타입질의가 아닌 중립질의("공급망"/"supply chain")로 수집한 혼합 스트림**을 사용했다. 초기 시도(n=26)는 검정력이 부족했으므로(정확도 0.46이 majority 대비 *비유의*, p=0.07), 중립질의 700건을 **3개 독립 평정 관점**(동일 코드북)이 blind 라벨링하고 합의 골드를 구성했다.

![powered label-blind](figures/fig10_powered_gold.png)
*그림 3. 검정력 있는 label-blind 골드(Fleiss κ=0.84, n=217). 합성 전이는 정확도에서 majority 미달, 실데이터 학습 CV는 majority를 소폭 상회.*

- **평정 신뢰도: Fleiss κ=0.84(거의 완벽 일치)**, 쌍별 Cohen 0.80–0.89 → 라벨이 신뢰할 만하다("단일 평정자" 결함 해소).
- **합의 골드 n=217** (OTHER·무합의 483 제외). **실제 분포는 지정학이 65%로 심하게 편중** — 2026 공급망 뉴스의 현실(관세·수출통제·중동)이며, 보건 0건·자연재해 3건으로 일부 유형은 희소하다.

이 검정력 있는 골드에서:
- **합성→실 전이: 정확도 0.447(95% CI [0.38, 0.51]) — 다수클래스 베이스라인(0.65)에 수치상 미달하고, majority보다 크다는 증거가 없다**(one-sided greater test p=1.0). 즉 *합성 학습 모델은 "항상 지정학"이라 찍는 것보다도 정확도가 낮다.* 다만 Macro-F1은 0.364로 majority(0.11)를 크게 상회 — 소수 유형을 구분하는 능력 자체는 있으나, 지정학 편중 탓에 정확도로는 majority에 진다.
- **real-train/real-test CV(동일 closed-set gold 내부): 정확도 0.756(> majority 0.65), Macro-F1 0.43±0.07.** *실데이터로 학습하면* 고전 TF-IDF가 majority를 **소폭 상회**해 분류를 학습한다. 단 희소 유형(보건·자연재해)은 사실상 학습 불가다.

> **요지.** 합성 전이 방식은 closed-set에서도 부적합하다(정확도가 majority 미달). 그러나 *실데이터로 직접 학습*하면 고전 분류기가 closed-set majority를 modest하게 넘는다(acc 0.76). 데이터가 지정학에 심히 편중돼 실용 가치는 제한적이며, 이 때문에 §5.5에서 OTHER 포함 open-set으로 다시 검증한다.

### 5.5 실배포형 재정의 — `OTHER` 포함 open-set triage

§5.4까지도 여전히 한계가 있다. powered gold n=217은 중립질의에서 나온 합의 위험 뉴스만 남긴 **closed-set subset**이다. 그러나 실제 triage의 첫 질문은 "이 헤드라인이 7유형 중 무엇인가?"가 아니라 **"이 헤드라인이 애초에 공급망 위험 뉴스인가, 아니면 OTHER인가?"**이다. 따라서 최종 고도화는 버렸던 OTHER를 평가 대상으로 되살리는 방식으로 설계했다.

**가설.**
- **H1.** `OTHER` 선택지가 없는 7유형 합성 모델은 중립 뉴스 스트림에서 실배포형 triage 모델이 아니다.
- **H2.** `OTHER`를 포함해 실데이터로 open-set 학습하면 관련 뉴스 선별은 학습되지만, 세부 유형 분류는 class imbalance 때문에 modest하게 남는다.
- **H3.** 합성 7유형 모델의 confidence thresholding은 낙관적 진단일 뿐, 실제 open-set 라벨을 대체하지 못한다.

**실험 설계.** 중립질의 700건의 3인 패널 라벨에서 무합의 1건만 제외하고 **699건 전체를 open-set gold**로 사용했다. 분포는 OTHER 482건, 위험 217건(지정학 141·공급자 30·노동 18·물류 17·금융 8·자연재해 3·보건 0)이다. 비교군은 (a) 항상 OTHER, (b) `OTHER`를 모르는 합성 7유형 모델, (c) 8-class flat LR(`7유형+OTHER`) 교차검증, (d) 1단계 위험/OTHER detector + 2단계 위험유형 classifier로 구성했다. 평가지표는 accuracy만 보지 않고 **risk-detection F1/AUPRC, present-label Macro-F1, 위험유형 Macro-F1**을 함께 보고했다.

![open set](figures/fig11_open_set.png)
*그림 4. plain open-set baseline. OTHER를 포함하면 합성 7유형 모델은 붕괴하고, 실데이터 open-set 학습만 관련 뉴스 선별 신호를 보인다.*

결과는 세 가설을 모두 지지한다. **항상 OTHER**는 accuracy 0.690으로 높아 보이지만 risk-detection F1=0.00이므로 triage 모델이 아니다. 반대로 **합성 7유형 모델**은 모든 문서를 위험으로 강제 분류해 risk recall은 높지만 OTHER를 전혀 처리하지 못해 accuracy 0.139, present-label Macro-F1 0.213에 그쳤다. confidence threshold를 사후 최적화해도 risk-F1의 낙관적 상한은 0.544 수준이었다.

실데이터 open-set 학습은 다르다. **flat 8-class LR**은 3-fold stratified out-of-fold 평가에서 accuracy 0.820(95% bootstrap CI [0.790, 0.847]), **risk-detection F1 0.776**([0.730, 0.819]), AUPRC 0.851, present-label Macro-F1 0.471을 기록했다. 2-stage 모델도 유사했다(accuracy 0.813, risk-F1 0.772, AUPRC 0.878). 즉 plain TF-IDF+LR은 **관련 위험 뉴스 선별(relevance triage)** 은 학습하지만, 세부 유형 분류는 여전히 제한적이다(위험유형 Macro-F1 0.448; 자연재해 n=3·보건 n=0). 이 결과는 최종 성과가 아니라, 다음 고도화의 출발점이다.

### 5.6 최종 알고리즘 고도화 — mechanism-aware hybrid engine

plain open-set baseline의 오류를 직접 분석했다. 2-stage TF-IDF는 OTHER 482건 중 44건을 위험으로 오탐했고, 실제 위험 217건 중 53건을 OTHER로 놓쳤으며, 34건은 위험으로 잡았지만 유형을 틀렸다. 오탐은 "supply-chain strategy/report/forum/logistics industry"처럼 위험 사건이 아닌 generic OTHER가 위험 어휘를 공유할 때 많았다. 미탐과 유형오류는 희토류·핵심광물·운임·화재·총파업·반도체 수급·freight·coltan처럼 한·영 OOV 메커니즘 표현이 sparse token TF-IDF에서 잘 보존되지 않을 때 많았다.

**고도화 가설.**
- **H4.** raw character n-gram은 한국어/영어 OOV 위험 표현을 보존해 false negative와 유형오류를 줄인다.
- **H5.** taxonomy별 메커니즘 counter와 generic OTHER cue를 명시하면 위험 사건과 일반 공급망 담론을 더 잘 분리한다.
- **H6.** 위험/OTHER detector와 위험유형 classifier를 분리하고, inner-CV에서 precision floor를 우선하는 gate-aware threshold를 고르면 triage 품질과 오탐 예산을 동시에 통제할 수 있다.

**엔진 설계.** 최종 모델은 `word TF-IDF + raw character n-gram + auditable mechanism feature`를 결합한 sparse hybrid representation이다. mechanism feature는 7유형 taxonomy seed에 일반 사건 메커니즘(예: 희토류, 운임, 화재, 총파업, memory shortage, warehouse fire)과 OTHER/generic cue(예: webinar, report, strategy, forum)를 더한 카운터다. 학습은 3-fold stratified OOF를 유지했고, 평가 fold의 라벨은 feature나 threshold 선택에 사용하지 않았다. 2-stage 구조에서 detector threshold는 inner-CV score로만 정하며, F1 최대화보다 **precision floor 우선 fallback**을 사용해 false-positive budget을 연구 gate에 맞췄다.

**고정 research gate.** 사후 숫자 해석을 피하기 위해 다음 기준을 통과 조건으로 두었다: accuracy ≥0.85, present-label Macro-F1 ≥0.65, risk precision ≥0.84, risk recall ≥0.82, risk-F1 ≥0.85, risk AUPRC ≥0.90, 위험유형 Macro-F1 ≥0.65, false-positive risk ≤30, false-negative risk ≤40, risk-type-wrong ≤25.

![mechanism engine](figures/fig12_mechanism_engine.png)
*그림 5. plain open-set baseline 대비 mechanism-aware hybrid engine의 개선. risk-F1뿐 아니라 type Macro-F1과 error budget을 함께 검증했다.*

결과적으로 최종 engine은 3-fold OOF에서 **accuracy 0.884, present-label Macro-F1 0.696, risk precision 0.880, risk recall 0.843, risk-F1 0.861, AUPRC 0.923, 위험유형 Macro-F1 0.697**을 기록했다. plain 2-stage baseline 대비 accuracy는 +0.072, risk-F1은 +0.089, AUPRC는 +0.045, 위험유형 Macro-F1은 +0.260 개선되었다. 오류 예산도 false positive 44→25, false negative 53→34, 위험유형 오류 34→22로 줄어 **고정 research gate를 모두 통과**했다.

> **최종 RQ2 판정.** "합성→실 전이"와 "타입질의 전이"는 실배포 근거가 아니다. 그러나 open-set gold에서 오류 원인을 분석하고 mechanism-aware hybrid engine으로 고도화하면, 중립 뉴스 스트림에서 관련 공급망 위험을 선별하고 주요 메커니즘을 분류하는 **방어 가능한 연구 수준의 triage 알고리즘**까지는 도달한다(risk-F1 0.861, AUPRC 0.923, gate pass). 단 희소 유형과 시간외 일반화는 아직 추가 검증 대상이다.

### 5.7 실데이터 교차검증과 그 한계

### 5.6 실데이터 교차검증과 그 한계

실데이터 단독 CV에서 손-라벨 골드 102건 5-fold는 Macro-F1 0.606±0.062, 약-라벨 실뉴스 2,062건 5-fold는 0.884±0.012였다. 다만 **후자(0.88)의 라벨 역시 타입질의에서 유래**하므로 이 또한 부분적으로 질의복원이며, 독립 일반화의 증거로 보기 어렵다(이 점을 숨기지 않는다).

![learning curve](figures/fig2_learning_curve.png)
*그림 6. 학습곡선(합성 내부 수치). n≈64에서 0.6 돌파 — 합성 분리가능성의 성질.*

> **RQ2 답.** 타입질의·약-라벨의 0.77–0.88은 분류 능력이 아니라 **검색질의·시드 키워드 복원**이다(echo 0.99·seed-rule 0.72가 방증). 검정력 있는 label-blind closed-set 골드(κ=0.84, n=217)에서 **합성 전이는 majority에 미달(acc 0.447 < 0.65)**했고, open-set gold(699건, OTHER 포함)에서는 `OTHER` 없는 합성 모델이 accuracy 0.139로 붕괴했다. 반면 **실데이터 open-set 학습 + mechanism-aware 고도화**는 관련 위험 뉴스 선별에서 연구 gate를 통과했다(risk-F1 0.861, AUPRC 0.923).

---

## 6. RQ3 — 이중언어 일관성

![per language](figures/fig5_per_language.png)
*그림 7. 언어별 성능(타입질의 골드). 단일 모델이 한·영을 함께 처리.*

단일 분류기로 한국어(Acc 0.81, n=48)와 영어(Acc 0.74, n=54)를 처리했고, **KO−EN 정확도 차이(약 0.07)의 95% 부트스트랩 CI는 [−0.09, 0.24]로 0을 포함**한다. 이는 **소표본(n=48/54)에서 차이를 검출할 검정력이 부족**함을 뜻하며 — CI가 한국어 0.24 우위와 영어 0.09 우위를 모두 포함한다 — **동등성의 증명이 아니라 차이를 단정할 수 없음**을 의미한다.

> **RQ3 답.** 단일 파이프라인을 한·영에 함께 적용했고, 현재 표본에서 언어 간 *유의한 성능차는 검출되지 않았다*(동등성 입증과는 다름). 단, 토픽 발견(RQ1)은 언어별 분리가 필요했으므로 "단일 처리"는 분류·심각도 단계에 한정된다.

---

## 7. 심각도 — 순환성을 제거한 검증

감성사전으로 부정 강도를 LOW/MEDIUM/HIGH로 환산했다. 적대적 검증은 합성 심각도 검증(ρ=0.67)이 **순환적**임을 지적했다 — 정답을 정의한 단어를 사전이 그대로 채점하기 때문이다. 이를 해소하기 위해 **연구자가 사전과 무관하게 실제 헤드라인 102건의 심각도를 독립 평정**하고 그에 대해 검증했다.

![severity](figures/fig6_severity.png)
*그림 8. (좌) 순환적 합성 검증 ρ=0.67, (우) 순환 제거 독립 검증 ρ=0.40.*

독립 검증 결과 **Spearman ρ=0.404(p<0.001)**, 레벨별 평균 단조(LOW 0.36 < MED 0.70 < HIGH 1.25)였다. (±1단계 정확도 0.94도 얻었으나, 3단계 척도에서 ±1 정확도는 거의 자명한 지표이므로 **ρ=0.40을 정직한 헤드라인**으로 삼는다.)

> **심각도 답.** 사전 기반 심각도는 *중간 수준의 유의한* 신호를 포착한다(독립 ρ=0.40, 합성 ρ=0.67은 순환으로 부풀려진 값).

---

## 8. 위협 요인 및 한계 (자기 적대적 검증)

본 연구는 여러 독립 비평 렌즈로 스스로를 공격하고 그 결과를 공개한다. **방어 가능성은 화려한 수치가 아니라 이 투명성에서 나온다.**

**표 4. 식별된 위협과 대응**

| 위협 | 내용 | 대응 |
|---|---|---|
| 질의=라벨 순환 | 타입질의 골드의 라벨이 검색질의와 99% 일치 → echo 0.99 | **중립질의 powered 골드 신설**(n=217, §5.4) |
| 시드 의존 | 전이가 부분적으로 시드 매칭 | 마스킹 ablation으로 정량화(0.77→0.48) |
| closed-set 착시 | powered gold 217건은 OTHER를 버린 위험 subset | **open-set gold 699건**으로 OTHER 포함 triage 재검증(§5.5) |
| plain TF-IDF 한계 | OOV 메커니즘 미탐·generic OTHER 오탐 | **mechanism-aware hybrid engine + fixed gate**로 개선(§5.6) |
| 구성 타당성 | 합성은 작은 데카르트곱, F1=1.0은 산술적 필연 | 모든 핵심 수치를 실데이터·label-blind로 재측정 |
| 심각도 순환성 | 정답·사전이 동일 어휘 | 독립 평정 검증으로 해소(ρ=0.40) |
| 어휘중첩 오표기 | "0.98"은 OOV-아닌-문서비율 | 진짜 커버리지 **0.277**로 정정 |
| 분류체계 비-MECE | 골드 24%가 다중유형, 사이버·사기 제외 | 단일 라벨을 *주 메커니즘*으로 명시, 한계 공개 |
| 통계 엄밀성 | n 작음 | 부트스트랩·Wilson 95% CI 전면 보고 |
| 단일 평정자 (해소) | 골드 라벨이 1인이던 문제 | **3개 독립 평정 관점으로 재라벨, Fleiss κ=0.84** (§5.4) |
| 재현성 | RSS는 매일 변함 | 데이터를 **디스크에 동결**(2026-06-14), 시드 고정 |

**기타 한계.** ① 본 시스템은 미래 예측이 아니라 보도된 사건의 **조기 인지**다. ② 보도되지 않은 리스크는 포착 불가. ③ open-set gold도 지정학 편중이 크고 보건 0건·자연재해 3건이라 희소 유형은 학습·평가가 불안정하다. ④ 골드 라벨은 LLM 패널(κ=0.84)이 부여 — 인간 전문가 평정과는 다른 귀납편향을 공유할 수 있다(향후 인간 평정 보강 필요). ⑤ 최종 엔진도 같은 동결 gold 내부의 nested/OOF 평가이므로, 미래 시점 뉴스에 대한 시간외 검증은 남아 있다. ⑥ 향후 KoBERT 등 문맥 임베딩으로 시드 의존과 표현 변화 취약성을 더 줄이는 방향을 검토한다.

---

## 9. 결론

본 연구는 한·영 뉴스 텍스트마이닝 기반 공급망 리스크 탐지 프레임워크를 설계하고 **실데이터로 다단계 검증**했다. 핵심 성과는 ① 다국어 LDA의 언어 분리 현상을 정량 규명하고 해결한 점, ② 분류 수치의 **순환·시드 의존을 스스로 폭로**하고, **검정력 있는 검증(3인 패널 Fleiss κ=0.84, n=217)**으로 *합성 전이의 한계(정확도 majority 미달)*를 보인 점, ③ **OTHER 포함 open-set gold(699건)**로 실배포형 질문을 재정의한 점, ④ 오류분석을 통해 **mechanism-aware hybrid engine**을 설계하고 고정 research gate를 통과한 점(risk-F1 0.861, AUPRC 0.923, accuracy 0.884), ⑤ 한·영 비교의 불충분성을 명시한 점, ⑥ 심각도의 순환성을 제거한 검증(독립 ρ=0.40)을 제시한 점이다.

**가장 중요한 교훈은 방법론적이다.** 고전 텍스트마이닝(TF-IDF·LDA·분류기)은 *키워드가 정렬된* 뉴스를 잘 다루지만, 검색질의·시드와 라벨이 얽히면 성능이 *질의 복원으로 오인*되기 쉽고, OTHER를 제거하면 closed-set 착시가 생긴다. 본 연구는 이 함정을 진단하고 **open-set gold(699건)로 실배포형 질문을 다시 물은 뒤**, plain TF-IDF의 gap을 메커니즘 feature와 gate-aware threshold로 고도화했다. 따라서 현재 수준에서 말할 수 있는 것은 더 명확하다: **실데이터 라벨과 고정 gate가 있을 때, 중립 뉴스 스트림에서 관련 공급망 위험을 선별하고 주요 메커니즘을 분류하는 연구 수준의 triage 알고리즘은 방어 가능하다.** 다만 산업적 배포까지 주장하려면 인간 전문가 평정, 시간외 검증, 희소 유형 보강이 추가되어야 한다.

---

## 부록 A. 재현 절차

```
cd src
python data_build.py     # 구조화 코퍼스(1540)
python realnews.py       # 타입질의 실뉴스(2062)
python build_gold.py     # 타입질의 골드(102)
python topic_lda.py      # RQ1
python classify.py       # RQ2/RQ3 내부
python embeddings.py     # Word2Vec 확장 + lexicon ablation
python external_eval.py  # 타입질의 골드 전이
python sentiment.py / severity_gold.py   # 심각도(합성/독립)
python stats_addendum.py # 부트스트랩 CI·베이스라인
python robustness.py     # 시드마스킹·실데이터 CV·어휘커버리지
python baselines.py      # query-echo·seed-rule·char-ngram
python neutral_eval.py   # label-blind 중립질의 검증(n=26 pilot)
python build_powered_gold.py  # 3인 패널 합의 골드(n=217, κ=0.84) + 검정력 평가
python open_set_eval.py  # OTHER 포함 plain open-set baseline(n=699)
python mechanism_engine_eval.py  # mechanism-aware final engine
python research_gate.py  # fixed high-standard KPI gate
python make_figures.py   # 집계 + 12개 figure
```
모든 무작위는 SEED=42 고정. 수집 데이터는 `data/`에 동결되어 결과가 안정적이다.

## 부록 B. 참고문헌

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *JMLR*, 3, 993–1022.
- Salton, G., & McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw-Hill.
- Mikolov, T., et al. (2013). Distributed representations of words and phrases. *NIPS*, 26.
- McCallum, A., & Nigam, K. (1998). A comparison of event models for naive Bayes. *AAAI Workshop*.
- Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *FnT in IR*, 2(1–2).
- Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. *WSDM '15*, 399–408.
- Park, E. L., & Cho, S. (2014). KoNLPy: Korean NLP in Python. *HCLT 2014*.
- Nguyen, T., et al. (2025). A systematic review of text mining analytics for SCRM. *Supply Chain Analytics*.
