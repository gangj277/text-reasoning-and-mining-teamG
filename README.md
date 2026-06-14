# 한·영 뉴스 텍스트마이닝 기반 공급망 리스크 탐지 시스템
### Team G ｜ 지식추론과 텍스트마이닝 Term Project

한국어·영어 뉴스를 단일 파이프라인에서 분석해 공급망 리스크를 **탐지·7유형 분류·심각도 평가**하는
고전 텍스트마이닝(TF-IDF·LDA·Word2Vec·감성사전·지도분류기) 프레임워크. **LLM API 미사용**, 로컬 CPU, 100% 재현.

## 최종 산출물
- **`report/TeamG_Final_Report.pdf`** — 최종 결과 보고서 (한글, figure 포함)
- `results/figures/*.png` — 12개 핵심 figure
- `results/metrics.json` — 전 결과 집계
- `results/tables/*.csv` — 토픽·분류·확장 사전 표

## 핵심 결과 (정직한 그림)
| 항목 | 수치 | 의미 |
|---|---|---|
| RQ1 naive LDA 언어순도 | 0.86 | 통합 LDA는 언어로 분리됨(→ 언어별 학습) |
| RQ2 내부 합성 F1 | 1.00 | 과적합 진단용(실세계 추정 아님) |
| RQ2 타입질의 골드 F1 | 0.766 | 단, query-echo 0.99가 능가 → 질의복원 |
| RQ2 powered closed-set | 합성전이 acc 0.447 < majority 0.650 | 217개 risk 합의건만 남긴 폐쇄형 검증 |
| RQ2 plain open-set triage | risk-F1 0.776 / acc 0.820 | OTHER 포함 699건에서 실데이터 학습 필요성 확인 |
| RQ2 **mechanism-aware engine** | **risk-F1 0.861 / acc 0.884 / AUPRC 0.923** | **문자 n-gram+메커니즘 feature+gate-aware threshold가 fixed research gate 통과** |
| RQ3 KO−EN 차이 | 유의하지 않음 | 단일 파이프라인 언어 동등 (95% CI 0 포함) |
| 심각도 독립 ρ | 0.40 (p<0.001) | 순환 제거 검증(합성 0.67은 순환) |

핵심 메시지: 고전 텍스트마이닝은 *키워드 정렬* 뉴스와 폐쇄형 7유형 분류만으로는 실배포 주장을 할 수 없고,
`OTHER`까지 포함한 open-set triage를 명시적으로 학습해야 관련 뉴스 선별 신호가 나타난다. 여기서 끝내지 않고
오류 분석으로 드러난 OOV 메커니즘·generic OTHER 혼동을 **mechanism-aware hybrid engine**으로 고도화하여
고정 KPI gate(accuracy/F1/AUPRC/error-budget)를 통과했다.

## 실행
```bash
# 의존: python3, gensim konlpy nltk scikit-learn scipy matplotlib markdown (+ Java17 for Okt)
cd src
python data_build.py && python realnews.py && python build_gold.py
python topic_lda.py && python classify.py && python embeddings.py && python external_eval.py
python sentiment.py && python severity_gold.py
python stats_addendum.py && python robustness.py && python baselines.py && python neutral_eval.py
python build_powered_gold.py && python open_set_eval.py
python mechanism_engine_eval.py && python research_gate.py
python make_figures.py && python render_report.py
```
모든 무작위 SEED=42 고정. `data/`는 동결 스냅샷(2026-06-14)이라 재실행 없이도 결과 안정.

## 모듈 구성 (`src/`)
`config`(분류체계) · `data_build`(합성 코퍼스) · `realnews`(실뉴스 RSS) · `build_gold`(타입질의 골드) ·
`preprocess`(Okt/NLTK) · `topic_lda`(RQ1) · `classify`(RQ2/3) · `embeddings`(Word2Vec) ·
`external_eval`(전이) · `sentiment`/`severity_gold`(심각도) · `stats_addendum`(CI) · `robustness`(시드마스킹/실CV) ·
`baselines`(echo/규칙) · `neutral_eval`(label-blind pilot) · `build_powered_gold`(3패널 합의) ·
`open_set_eval`(OTHER 포함 plain open-set baseline) · `mechanism_engine`/`mechanism_engine_eval`(최종 고도화 엔진) ·
`research_gate`(고정 KPI gate) · `make_figures` · `render_report`
