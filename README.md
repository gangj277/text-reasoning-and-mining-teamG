# Team G · Bilingual SCRM News Mining

> 한·영 뉴스 기반 공급망 리스크 탐지 연구  
> 지식추론과 텍스트마이닝 Term Project

GitHub: <https://github.com/gangj277/text-reasoning-and-mining-teamG>

## 1. What This Repository Contains

이 저장소는 한국어·영어 공급망 뉴스를 대상으로 **open-set SCRM triage engine**을 설계하고 검증한 최종 제출 패키지입니다. 핵심 질문은 단순히 “뉴스를 7개 리스크 유형으로 분류할 수 있는가”가 아니라, 실제 뉴스 스트림에서 먼저 **위험 기사와 OTHER 기사를 분리하고**, 위험 기사에만 SCRM 리스크 유형을 부여할 수 있는가입니다.

최종 산출물은 아래에 들어 있습니다.

| Path | Description |
|---|---|
| `scrm-news-mining-deck/deck-export.pdf` | 최종 발표 덱, 16 pages |
| `scrm-news-mining-deck/index.html` | HTML 기반 발표 덱 뷰어 |
| `report/TeamG_Final_Report.pdf` | 최종 보고서 PDF |
| `report/report.md` | 보고서 원문 Markdown |
| `src/` | 데이터 구축, 실험, 엔진, 평가, 리포트 렌더링 코드 |
| `data/` | 동결 데이터 스냅샷 |
| `results/` | 실험 결과 JSON, CSV, figure |
| `tests/` | mechanism engine 및 research gate 단위 테스트 |

## 2. Research Framing

공급망 충격은 재고, 리드타임, 매출 같은 정형 지표보다 뉴스 텍스트에 먼저 나타납니다. 하지만 실제 뉴스 스트림은 다음 세 조건 때문에 단순 분류 문제가 아닙니다.

1. **조기성**: 홍해, 희토류, 지진, 파업 같은 외생 충격은 정형 KPI보다 뉴스에 먼저 기록됩니다.
2. **희소성**: 중립 질의 기반 open-set gold 699건 중 위험 기사는 217건이고, OTHER는 482건입니다. 공급망 관련 기사 대부분은 실제 위험이 아닙니다.
3. **이질성**: 한·영 혼재, OOV 표현, generic supply-chain 담론이 섞여 있어 키워드나 closed-set 7유형 분류만으로는 실제 triage를 설명하기 어렵습니다.

따라서 본 연구는 최종 문제를 다음과 같이 정의합니다.

```text
Input  : headline + snippet from bilingual supply-chain news stream
Stage 1: risk detector      -> risk vs OTHER
Stage 2: risk type classifier -> one of 7 SCRM risk types, only if risk
Output : risk decision, risk type, confidence, evaluation metrics
```

## 3. Data Assets

| Asset | Size | Role |
|---|---:|---|
| Synthetic corpus | 1,540 | 통제된 학습·sanity-check 데이터 |
| Real news stream | 2,062 | Google News RSS 기반 실뉴스 코퍼스 |
| Typed-query gold | 102 | 초기 전이 검증 및 query-circularity 진단 |
| Powered risk gold | 217 | 중립 질의에서 합의된 위험 기사 subset, Fleiss κ=0.843 |
| Open-set gold | 699 | 최종 triage 평가셋, risk 217 / OTHER 482 |

모든 데이터와 결과는 저장소에 포함된 동결 스냅샷 기준입니다. 주요 평가 결과는 `results/*.json`과 `results/tables/*.csv`에 저장되어 있습니다.

## 4. Final Algorithm

최종 엔진은 `src/mechanism_engine.py`에 구현된 **mechanism-aware hybrid open-set engine**입니다.

Representation:

```text
φ(xᵢ) = [ψ_word(tokᵢ); ψ_char(rawᵢ); m_tax(xᵢ); c_other(xᵢ)]
```

구성 요소:

- `ψ_word`: normalized token TF-IDF, 1-2 gram, `min_df=2`
- `ψ_char`: raw title `char_wb` n-gram, 2-5 gram, max 8,000 features
- `m_tax`: 7개 SCRM taxonomy-level mechanism counter
- `c_other`: OTHER/generic/cyber meta cue

Decision rule:

```text
sᵢ = P(zᵢ = risk | φ(xᵢ))
ŷᵢ = OTHER                         if sᵢ < τₖ
ŷᵢ = argmax_r P(r | φ(xᵢ), risk)   otherwise
```

`τₖ`는 evaluation fold를 보지 않고 training fold 내부 OOF score로 calibration합니다. 최종 평가는 3-fold OOF 예측만 합산합니다.

## 5. Key Results

| Experiment | Result | Interpretation |
|---|---:|---|
| Naive bilingual LDA language purity | 0.856 | 통합 LDA는 리스크 유형보다 언어 경계를 먼저 분리 |
| Typed-query gold Macro-F1 | 0.766 | query-echo baseline 0.990에 미달, 질의순환 진단 |
| Powered risk gold synthetic transfer | acc 0.447 | majority acc 0.650 미달, synthetic closed-set 전이 한계 |
| Plain open-set two-stage baseline | risk-F1 0.772 / AUPRC 0.878 | 관련성 신호는 있으나 gate 실패 |
| Mechanism-aware engine | acc 0.884 / risk-F1 0.861 / AUPRC 0.923 | fixed research gate 통과 |
| Risk type macro-F1 | 0.697 | risk로 통과한 기사에 대한 7유형 typing 성능 |
| Independent severity validation | Spearman ρ=0.404, p<0.001 | 순환 제거 후에도 유의한 단조 상관 |

최종 research gate는 `results/research_gate.json`에 기록되어 있습니다.

```json
{
  "passed": true,
  "observed": {
    "accuracy_8way": 0.8841,
    "risk_f1": 0.8612,
    "risk_auprc": 0.9231,
    "false_positive_risk": 25,
    "false_negative_risk": 34,
    "risk_type_wrong": 22
  }
}
```

## 6. How To Reproduce

The repository includes frozen data and generated results. To inspect the final artifacts, no rerun is required.

To rerun the full pipeline:

```bash
cd src

python data_build.py
python realnews.py
python build_gold.py
python topic_lda.py
python classify.py
python embeddings.py
python external_eval.py
python sentiment.py
python severity_gold.py
python stats_addendum.py
python robustness.py
python baselines.py
python neutral_eval.py
python build_powered_gold.py
python open_set_eval.py
python mechanism_engine_eval.py
python research_gate.py
python make_figures.py
python render_report.py
```

Expected environment:

- Python 3.10+
- Java runtime for KoNLPy Okt
- Python packages: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `gensim`, `nltk`, `konlpy`, `markdown`

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## 7. Repository Structure

```text
.
├── data/                       # frozen source and gold data
├── results/                    # metrics, prediction tables, figures
│   ├── figures/
│   └── tables/
├── report/                     # final report PDF, HTML, Markdown
├── scrm-news-mining-deck/      # final HTML deck and exported PDF
├── src/                        # research pipeline and final engine
└── tests/                      # regression tests for engine and gate
```

Core modules:

- `src/config.py`: 7-class SCRM taxonomy and shared constants
- `src/preprocess.py`: Korean/English preprocessing
- `src/topic_lda.py`: RQ1 topic modeling
- `src/classify.py`, `src/external_eval.py`: early classification experiments
- `src/baselines.py`, `src/neutral_eval.py`: circularity and neutral-query diagnostics
- `src/open_set_eval.py`: OTHER-included open-set baseline
- `src/mechanism_engine.py`: final mechanism-aware triage engine
- `src/mechanism_engine_eval.py`: OOF evaluation for final engine
- `src/research_gate.py`: fixed KPI gate evaluation
- `src/make_figures.py`, `src/render_report.py`: final report artifacts

## 8. Submission Notes

- The final deck is `scrm-news-mining-deck/deck-export.pdf`.
- The final report is `report/TeamG_Final_Report.pdf`.
- The code path for the final research contribution is `src/mechanism_engine.py` → `src/mechanism_engine_eval.py` → `src/research_gate.py`.
- The archived 25-slide draft under `scrm-news-mining-deck/archive-25slide-draft-20260614-1500/` is retained only for provenance and is not the final deck.

