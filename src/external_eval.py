"""
external_eval.py — 외부 타당성 검증 (본 연구의 핵심 실험).

합성 코퍼스로 학습한 분류기를, 연구자가 손-라벨링한 '실제' 뉴스 헤드라인 골드셋에
적용해 일반화 성능을 측정한다. 내부 합성 테스트(F1=1.0)는 합성 분리가능성의 산물일 뿐
이며, 진짜 질문 — "이 방법론이 실제 뉴스의 신호를 포착하는가" — 은 외부 골드셋이 답한다.

비교축:
  - 내부 합성 held-out F1 (상한, 과대)  vs  외부 실데이터 골드 F1 (실제 일반화)
  - 무작위 기대치(1/7=0.143) 대비 얼마나 우월한가 → leakage 가 아니라 실제 학습임을 입증
결과: results/rq_external.json, results/tables/external_confusion.csv
"""
from __future__ import annotations
import json, os, csv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from config import RESULTS_DIR, TABLE_DIR, RISK_CODES, GOLD_PATH, CODE2KO
from preprocess import load_jsonl, tokenize
from classify import load_xy, train_final_model


def run_external():
    # 1) 전체 합성 코퍼스로 최종 모델 학습
    X, y, _ = load_xy()
    vec, clf = train_final_model(X, y)

    # 2) 실데이터 골드셋 전처리 → 예측
    gold = load_jsonl(GOLD_PATH)
    g_tokens = [" ".join(tokenize(g["title"], g["lang"])) for g in gold]
    Xg = vec.transform(g_tokens)
    yg = np.array([g["gold"] for g in gold])
    lang_g = np.array([g["lang"] for g in gold])
    pred = clf.predict(Xg)

    # OOV 진단: 골드 헤드라인이 학습 어휘와 얼마나 겹치나
    nz = (Xg.sum(axis=1).A1 > 0).mean()

    macro = float(f1_score(yg, pred, average="macro"))
    acc = float(accuracy_score(yg, pred))
    per_lang = {}
    for lg in ("ko", "en"):
        m = lang_g == lg
        if m.sum():
            per_lang[lg] = {"macro_f1": round(float(f1_score(yg[m], pred[m], average="macro")), 4),
                            "acc": round(float(accuracy_score(yg[m], pred[m])), 4), "n": int(m.sum())}

    rep = classification_report(yg, pred, labels=RISK_CODES, output_dict=True, zero_division=0)
    cm = confusion_matrix(yg, pred, labels=RISK_CODES)

    baseline = 1.0 / len(RISK_CODES)
    out = {"n_gold": len(gold), "vocab_overlap_rate": round(float(nz), 3),
           "external_macro_f1": round(macro, 4), "external_acc": round(acc, 4),
           "random_baseline_acc": round(baseline, 3),
           "lift_over_random": round(acc / baseline, 2),
           "per_lang": per_lang,
           "per_class_f1": {c: round(rep[c]["f1-score"], 3) for c in RISK_CODES if c in rep},
           "confusion": {"labels": RISK_CODES, "matrix": cm.tolist()}}
    with open(os.path.join(RESULTS_DIR, "rq_external.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(os.path.join(TABLE_DIR, "external_confusion.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [CODE2KO[c] for c in RISK_CODES])
        for i, c in enumerate(RISK_CODES):
            w.writerow([CODE2KO[c]] + cm[i].tolist())

    # 오분류 사례 일부 저장(정성 분석용)
    errs = [{"title": gold[i]["title"], "lang": gold[i]["lang"],
             "gold": yg[i], "pred": pred[i]} for i in range(len(gold)) if yg[i] != pred[i]]
    with open(os.path.join(RESULTS_DIR, "external_errors.json"), "w", encoding="utf-8") as f:
        json.dump(errs, f, ensure_ascii=False, indent=2)
    return out, errs


if __name__ == "__main__":
    print("=== 외부 타당성 검증: 합성학습 → 실데이터 골드 ===")
    out, errs = run_external()
    print(f"  골드 {out['n_gold']}건 | 어휘중첩률 {out['vocab_overlap_rate']}")
    print(f"  외부 Macro-F1 = {out['external_macro_f1']}")
    print(f"  외부 정확도   = {out['external_acc']}  (무작위 {out['random_baseline_acc']}, "
          f"{out['lift_over_random']}배)")
    print(f"  언어별: {out['per_lang']}")
    print(f"  유형별 F1: {out['per_class_f1']}")
    print(f"\n  오분류 {len(errs)}건 (예시 6):")
    for e in errs[:6]:
        print(f"   [{e['lang']}] gold={CODE2KO[e['gold']]} / pred={CODE2KO[e['pred']]} | {e['title'][:55]}")
