"""
make_figures.py — 전 결과 집계 + 출판급 figure 생성.

results/ 의 개별 rq*.json 을 읽어 master metrics.json 으로 통합하고,
12개 핵심 figure(PNG)를 results/figures/ 에 생성한다. (라벨은 폰트 안정성을 위해 ASCII)
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import RESULTS_DIR, FIG_DIR, METRICS_PATH, RISK_CODES

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
                     "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.3})
NAVY = "#22335a"; ORANGE = "#f5a623"; RED = "#d0021b"; GREEN = "#2e7d32"; GREY = "#9aa0a6"


def L(path):
    return json.load(open(os.path.join(RESULTS_DIR, path), encoding="utf-8"))


def main():
    rq1 = L("rq1_lda.json"); rq2 = L("rq2_classify.json")
    emb = L("rq2_embedding.json"); ext = L("rq_external.json"); sev = L("rq_severity.json")
    rob = L("robustness.json"); stt = L("stats_addendum.json"); sind = L("severity_independent.json")
    base = L("baselines.json"); neu = L("neutral_eval.json"); pw = L("powered_eval.json")
    opn = L("open_set_eval.json"); mech = L("mechanism_engine_eval.json")

    # ---------- master metrics ----------
    master = {
        "RQ1_topic_discovery": {
            "naive_combined_cv": rq1["naive_combined"]["cv"],
            "naive_lang_purity": rq1["naive_combined"]["lang_purity"],
            "stratified_cv_ko": rq1["stratified"]["ko"]["best_cv"],
            "stratified_cv_en": rq1["stratified"]["en"]["best_cv"],
            "best_k_ko": rq1["stratified"]["ko"]["best_k"],
            "best_k_en": rq1["stratified"]["en"]["best_k"],
            "recovered": rq1["headline_recovered"]},
        "RQ2_classification": {
            "internal_macro_f1": rq2["test_macro_f1"],
            "best_model": rq2["best_model"], "ablation": rq2["ablation"],
            "threshold_n_at_0.6": rq2["threshold_n_at_0.6"],
            "external_macro_f1": ext["external_macro_f1"],
            "external_acc": ext["external_acc"], "lift_over_random": ext["lift_over_random"],
            "lexicon_baseline_acc": emb["lexicon_external_acc_all"]},
        "RQ3_bilingual": {
            "internal_ko_f1": rq2["per_lang_f1"].get("ko"),
            "internal_en_f1": rq2["per_lang_f1"].get("en"),
            "external_ko": ext["per_lang"]["ko"], "external_en": ext["per_lang"]["en"]},
        "severity": {"synthetic_circular_rho": sev["spearman_rho"],
                     "independent_real_rho": sind["independent_spearman_rho"],
                     "independent_adjacent_acc": sind["adjacent_acc"],
                     "independent_mean_by_level": sind["mean_score_by_independent_level"]},
        "robustness": {
            "seed_mask_full_f1": rob["A_seed_mask_ablation"]["full"]["macro_f1"],
            "seed_mask_masked_f1": rob["A_seed_mask_ablation"]["seed_masked"]["macro_f1"],
            "pct_tokens_removed": rob["A_seed_mask_ablation"]["pct_tokens_removed"],
            "real_gold_cv_f1": rob["B_real_cv"]["gold_cv"]["macro_f1_mean"],
            "real_weaklabel_cv_f1": rob["B_real_cv"]["weaklabel_cv"]["macro_f1_mean"],
            "true_vocab_coverage": rob["D_vocab_coverage"]["type_token_coverage"]},
        "statistics": {
            "external_acc_ci95": stt["acc_boot_ci95"], "external_f1_ci95": stt["macro_f1_boot_ci95"],
            "majority_baseline_acc": stt["majority_baseline_acc"],
            "ko_minus_en_diff": stt["ko_minus_en_acc_diff"], "ko_en_ci95": stt["ko_minus_en_ci95"],
            "ko_en_diff_significant": stt["ko_en_diff_significant"]},
        "baselines_typed_gold": {
            "query_echo_f1": base["query_echo"]["macro_f1"],
            "full_pipeline_f1": base["full_pipeline"]["macro_f1"],
            "seed_rule_f1": base["seed_rule"]["macro_f1_covered"],
            "char_ngram_f1": base["char_ngram_no_preproc"]["macro_f1"],
            "majority_acc": base["majority"]["acc"]},
        "label_blind_neutral_gold": {
            "n": neu["n"], "full_acc": neu["full"]["acc"],
            "seed_masked_acc": neu["seed_masked"]["acc"],
            "majority_acc": neu["majority_baseline_acc"],
            "lift_over_majority": round(neu["full"]["acc"]/neu["majority_baseline_acc"], 2)},
        "powered_label_blind": {
            "n": pw["gold"]["n"], "fleiss_kappa": pw["annotation"]["fleiss_kappa"],
            "majority_acc": pw["gold"]["majority_acc"], "majority_class": pw["gold"]["majority_class"],
            "synth_transfer_acc": pw["transfer_synthetic_to_real"]["full_acc"],
            "synth_transfer_macro_f1": pw["transfer_synthetic_to_real"]["full_macro_f1"],
            "synth_acc_ci95": pw["transfer_synthetic_to_real"]["full_acc_ci95"],
            "synth_sig_vs_majority": pw["transfer_synthetic_to_real"]["sig_vs_majority"],
            "real_cv_acc": pw["real_cv_on_powered_gold"]["acc_mean"],
            "real_cv_macro_f1": pw["real_cv_on_powered_gold"]["macro_f1_mean"]},
        "open_set_triage": {
            "n": opn["data"]["consensus_n"],
            "risk_n": opn["data"]["risk_consensus_n"],
            "other_n": opn["data"]["other_consensus_n"],
            "risk_prevalence": opn["data"]["risk_prevalence"],
            "always_other_acc": opn["baselines"]["always_other"]["accuracy_8way"],
            "always_other_risk_f1": opn["baselines"]["always_other"]["risk_detection"]["f1"],
            "synthetic_forced_acc": opn["synthetic_transfer_open_stream"]["forced_7class_no_other"]["accuracy_8way"],
            "synthetic_forced_macro_f1_present": opn["synthetic_transfer_open_stream"]["forced_7class_no_other"]["macro_f1_present_labels"],
            "synthetic_forced_risk_f1": opn["synthetic_transfer_open_stream"]["forced_7class_no_other"]["risk_detection"]["f1"],
            "flat_cv_acc": opn["real_open_set_cv"]["flat_8class_lr"]["accuracy_8way"],
            "flat_cv_macro_f1_present": opn["real_open_set_cv"]["flat_8class_lr"]["macro_f1_present_labels"],
            "flat_cv_risk_f1": opn["real_open_set_cv"]["flat_8class_lr"]["risk_detection"]["f1"],
            "flat_cv_risk_auprc": opn["real_open_set_cv"]["flat_8class_lr"]["risk_detection"]["average_precision"],
            "flat_cv_risk_type_macro_f1_present": opn["real_open_set_cv"]["flat_8class_lr"]["risk_type_macro_f1_present_risk_labels"],
            "two_stage_cv_acc": opn["real_open_set_cv"]["two_stage_detector_then_typer"]["accuracy_8way"],
            "two_stage_cv_macro_f1_present": opn["real_open_set_cv"]["two_stage_detector_then_typer"]["macro_f1_present_labels"],
            "two_stage_cv_risk_f1": opn["real_open_set_cv"]["two_stage_detector_then_typer"]["risk_detection"]["f1"],
            "two_stage_cv_risk_auprc": opn["real_open_set_cv"]["two_stage_detector_then_typer"]["risk_detection"]["average_precision"],
            "two_stage_cv_risk_type_macro_f1_present": opn["real_open_set_cv"]["two_stage_detector_then_typer"]["risk_type_macro_f1_present_risk_labels"]},
        "mechanism_engine": {
            "acc": mech["mechanism_two_stage_cv"]["accuracy_8way"],
            "macro_f1_present": mech["mechanism_two_stage_cv"]["macro_f1_present_labels"],
            "risk_f1": mech["mechanism_two_stage_cv"]["risk_detection"]["f1"],
            "risk_auprc": mech["mechanism_two_stage_cv"]["risk_detection"]["average_precision"],
            "risk_type_macro_f1_present": mech["mechanism_two_stage_cv"]["risk_type_macro_f1_present_risk_labels"],
            "delta_acc": mech["comparison_to_plain_open_set"]["delta_two_stage"]["accuracy_8way"],
            "delta_macro_f1_present": mech["comparison_to_plain_open_set"]["delta_two_stage"]["macro_f1_present_labels"],
            "delta_risk_f1": mech["comparison_to_plain_open_set"]["delta_two_stage"]["risk_f1"],
            "delta_risk_auprc": mech["comparison_to_plain_open_set"]["delta_two_stage"]["risk_auprc"],
            "delta_risk_type_macro_f1_present": mech["comparison_to_plain_open_set"]["delta_two_stage"]["risk_type_macro_f1_present"]},
        "data": {"corpus_n": 1540, "real_news_n": 2062, "typed_gold_n": ext["n_gold"],
                 "neutral_blind_gold_n": neu["n"],
                 "oov_free_doc_rate": ext["vocab_overlap_rate"],
                 "true_vocab_coverage": rob["D_vocab_coverage"]["type_token_coverage"]},
    }
    json.dump(master, open(METRICS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---------- FIG 1: LDA coherence ----------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lg, col in (("ko", NAVY), ("en", ORANGE)):
        c = rq1["stratified"][lg]["coherence_by_k"]
        ks = sorted(int(k) for k in c); vals = [c[str(k)] for k in ks]
        ax.plot(ks, vals, "o-", color=col, label=f"{lg.upper()} (best K={rq1['stratified'][lg]['best_k']})")
        bk = rq1["stratified"][lg]["best_k"]
        ax.scatter([bk], [c[str(bk)]], s=160, facecolors="none", edgecolors=RED, linewidths=2, zorder=5)
    ax.axhline(rq1["naive_combined"]["cv"], ls="--", color=GREY,
               label=f"naive combined (lang-purity {rq1['naive_combined']['lang_purity']})")
    ax.set_xlabel("Number of topics K"); ax.set_ylabel("Topic coherence  c_v")
    ax.set_title("RQ1 — LDA coherence: language-stratified vs naive")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig1_lda_coherence.png")); plt.close(fig)

    # ---------- FIG 2: learning curve ----------
    lc = rq2["learning_curve"]
    ns = [c["n"] for c in lc]; mu = [c["f1_mean"] for c in lc]; sd = [c["f1_std"] for c in lc]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ns, mu, "o-", color=NAVY, lw=2)
    ax.fill_between(ns, np.array(mu)-np.array(sd), np.array(mu)+np.array(sd), color=NAVY, alpha=0.15)
    ax.axhline(0.6, ls="--", color=RED, label="practical threshold F1=0.6")
    thr = rq2["threshold_n_at_0.6"]
    if thr:
        ax.axvline(thr, ls=":", color=GREEN); ax.annotate(f"crosses 0.6\n at n≈{thr}",
                   (thr, 0.62), color=GREEN, fontsize=9)
    ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("Training set size (# articles)"); ax.set_ylabel("Macro-F1 (held-out)")
    ax.set_title("RQ2 — Learning curve (data scaling)"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig2_learning_curve.png")); plt.close(fig)

    # ---------- FIG 3: internal vs external (honesty figure) ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ["Internal\n(synthetic)", "External\n(real gold)", "Random\nbaseline"]
    vals = [rq2["test_macro_f1"], ext["external_macro_f1"], 1/7]
    cols = [GREY, NAVY, RED]
    b = ax.bar(bars, vals, color=cols, width=0.6)
    for r, v in zip(b, vals):
        ax.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Macro-F1")
    ax.set_title("Internal overfit vs typed-query transfer\n(see §5.2: this 0.77 is query-bucket recovery)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig3_internal_vs_external.png")); plt.close(fig)

    # ---------- FIG 4: external confusion ----------
    cm = np.array(ext["confusion"]["matrix"]); labels = ext["confusion"]["labels"]
    short = [c[:4] for c in labels]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(short, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()*0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (gold)")
    ax.set_title(f"RQ2 — External confusion (real news, n={ext['n_gold']})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig4_external_confusion.png")); plt.close(fig)

    # ---------- FIG 5: per-language (RQ3) ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    groups = ["Internal F1", "External F1", "External Acc"]
    ko = [rq2["per_lang_f1"]["ko"], ext["per_lang"]["ko"]["macro_f1"], ext["per_lang"]["ko"]["acc"]]
    en = [rq2["per_lang_f1"]["en"], ext["per_lang"]["en"]["macro_f1"], ext["per_lang"]["en"]["acc"]]
    x = np.arange(len(groups)); w = 0.35
    ax.bar(x-w/2, ko, w, color=NAVY, label="Korean")
    ax.bar(x+w/2, en, w, color=ORANGE, label="English")
    for i in range(len(groups)):
        ax.text(x[i]-w/2, ko[i]+0.01, f"{ko[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i]+w/2, en[i]+0.01, f"{en[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("RQ3 — Bilingual consistency (single pipeline)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig5_per_language.png")); plt.close(fig)

    # ---------- FIG 6: severity (circular synthetic vs independent real) ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    lv = ["LOW", "MEDIUM", "HIGH"]
    s_syn = sev["mean_score_by_true_level"]; s_ind = sind["mean_score_by_independent_level"]
    for ax, data, title, rho in (
        (a1, s_syn, f"Synthetic ground-truth\n(circular: ρ={sev['spearman_rho']})", sev['spearman_rho']),
        (a2, s_ind, f"Independent real ratings\n(de-circularized: ρ={sind['independent_spearman_rho']})", sind['independent_spearman_rho'])):
        vals = [data[k] for k in lv]
        b = ax.bar(lv, vals, color=[GREEN, ORANGE, RED], width=0.6)
        for r, v in zip(b, vals):
            ax.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
        ax.set_title(title, fontsize=10); ax.set_ylabel("Mean lexicon severity score")
    fig.suptitle("Severity validation — circular vs honest", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig6_severity.png")); plt.close(fig)

    # ---------- FIG 8: robustness (seed-mask ablation + real-data CV) ----------
    A = rob["A_seed_mask_ablation"]; B = rob["B_real_cv"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.3))
    names = ["Full\ntransfer", "Seed-\nmasked", "Random"]
    vals = [A["full"]["macro_f1"], A["seed_masked"]["macro_f1"], 1/7]
    b = a1.bar(names, vals, color=[NAVY, ORANGE, RED], width=0.62)
    for r, v in zip(b, vals):
        a1.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
    a1.set_ylim(0, 1.0); a1.set_ylabel("External Macro-F1")
    a1.set_title(f"Seed-mask ablation\n(remove {int(A['pct_tokens_removed']*100)}% tokens → still {A['seed_masked']['macro_f1']/(1/7):.1f}× random)", fontsize=10)
    rn = ["Synthetic→Real\n(transfer)", "Real gold CV\n(n=102)", "Real weak-label\nCV (n=2062)"]
    rv = [ext["external_macro_f1"], B["gold_cv"]["macro_f1_mean"], B["weaklabel_cv"]["macro_f1_mean"]]
    rerr = [0, B["gold_cv"]["macro_f1_std"], B["weaklabel_cv"]["macro_f1_std"]]
    b = a2.bar(rn, rv, yerr=rerr, capsize=4, color=[GREY, NAVY, GREEN], width=0.62)
    for r, v in zip(b, rv):
        a2.text(r.get_x()+r.get_width()/2, v+0.03, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
    a2.set_ylim(0, 1.0); a2.set_ylabel("Macro-F1")
    a2.set_title("Real-data evidence (no synthetic test)", fontsize=10)
    fig.suptitle("Robustness — is the signal real?", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig8_robustness.png")); plt.close(fig)

    # ---------- FIG 7: ablation ----------
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    names = ["Random", "W2V lexicon\n(unsupervised)", "TF-IDF + LR\n(supervised)"]
    vals = [1/7, emb["lexicon_external_acc_all"], ext["external_acc"]]
    b = ax.bar(names, vals, color=[RED, GREY, NAVY], width=0.6)
    for r, v in zip(b, vals):
        ax.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 0.95); ax.set_ylabel("External accuracy (real gold)")
    ax.set_title("Ablation — supervised learning adds value")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig7_ablation.png")); plt.close(fig)

    # ---------- FIG 9: honest baselines + label-blind generalization ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    names = ["query-echo\n(label=query)", "Full\npipeline", "seed-rule\n(keywords)",
             "char-ngram\n(no preproc)", "majority"]
    vals = [base["query_echo"]["macro_f1"], base["full_pipeline"]["macro_f1"],
            base["seed_rule"]["macro_f1_covered"], base["char_ngram_no_preproc"]["macro_f1"],
            base["majority"]["acc"]]
    cols = [RED, NAVY, ORANGE, GREY, "#cccccc"]
    b = a1.bar(names, vals, color=cols, width=0.66)
    for r, v in zip(b, vals):
        a1.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
    a1.set_ylim(0, 1.08); a1.set_ylabel("Macro-F1 / Acc")
    a1.set_title("Typed-query gold: trivial echo wins\n(→ 0.77 is query-bucket recovery)", fontsize=10)
    a1.tick_params(axis='x', labelsize=8)

    nn = ["Full\npipeline", "Seed-\nmasked", "Majority", "Random"]
    nv = [neu["full"]["acc"], neu["seed_masked"]["acc"], neu["majority_baseline_acc"], 1/7]
    b = a2.bar(nn, nv, color=[NAVY, ORANGE, "#cccccc", RED], width=0.62)
    for r, v in zip(b, nv):
        a2.text(r.get_x()+r.get_width()/2, v+0.015, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
    a2.set_ylim(0, 0.7); a2.set_ylabel("Accuracy")
    a2.set_title(f"Label-blind neutral gold (n={neu['n']})\nhonest generalization = {neu['full']['acc']:.2f} ({master['label_blind_neutral_gold']['lift_over_majority']}× majority)", fontsize=10)
    fig.suptitle("The honest picture: query-recovery vs true generalization", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig9_honest_baselines.png")); plt.close(fig)

    # ---------- FIG 10: powered label-blind gold (Fleiss kappa=0.84, n=217) ----------
    maj_acc = pw["gold"]["majority_acc"]; maj_f1 = 2*maj_acc/(1+maj_acc)/7  # majority macro-F1
    methods = ["Majority\nbaseline", "Synthetic→Real\n(transfer)", "Real-train CV\n(clean)"]
    accs = [maj_acc, pw["transfer_synthetic_to_real"]["full_acc"], pw["real_cv_on_powered_gold"]["acc_mean"]]
    f1s = [maj_f1, pw["transfer_synthetic_to_real"]["full_macro_f1"], pw["real_cv_on_powered_gold"]["macro_f1_mean"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(methods)); w = 0.38
    b1 = ax.bar(x-w/2, accs, w, color=NAVY, label="Accuracy")
    b2 = ax.bar(x+w/2, f1s, w, color=ORANGE, label="Macro-F1")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.012, f"{r.get_height():.2f}",
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.axhline(maj_acc, ls=":", color=GREY, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylim(0, 0.9); ax.set_ylabel("Score")
    ax.set_title(f"Powered label-blind gold (Fleiss κ={pw['annotation']['fleiss_kappa']}, n={pw['gold']['n']}, geopolitics-skewed)\n"
                 f"Synthetic transfer is BELOW majority on accuracy; real-train CV edges above", fontsize=10)
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig10_powered_gold.png")); plt.close(fig)

    # ---------- FIG 11: open-set triage (OTHER-inclusive deployment-like test) ----------
    flat = opn["real_open_set_cv"]["flat_8class_lr"]
    two = opn["real_open_set_cv"]["two_stage_detector_then_typer"]
    forced = opn["synthetic_transfer_open_stream"]["forced_7class_no_other"]
    always_other = opn["baselines"]["always_other"]

    methods = ["Always\nOTHER", "Synthetic 7-class\n(no OTHER)", "Real open-set\nflat LR", "Real open-set\ntwo-stage"]
    acc = [always_other["accuracy_8way"], forced["accuracy_8way"], flat["accuracy_8way"], two["accuracy_8way"]]
    risk_f1 = [always_other["risk_detection"]["f1"], forced["risk_detection"]["f1"], flat["risk_detection"]["f1"], two["risk_detection"]["f1"]]
    macro = [always_other["macro_f1_present_labels"], forced["macro_f1_present_labels"],
             flat["macro_f1_present_labels"], two["macro_f1_present_labels"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(methods)); w = 0.25
    bars = [
        ax1.bar(x - w, acc, w, color=GREY, label="Accuracy"),
        ax1.bar(x, risk_f1, w, color=NAVY, label="Risk-detection F1"),
        ax1.bar(x + w, macro, w, color=ORANGE, label="Macro-F1 (present labels)"),
    ]
    for group in bars:
        for r in group:
            ax1.text(r.get_x()+r.get_width()/2, r.get_height()+0.015, f"{r.get_height():.2f}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_ylim(0, 0.95)
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=8)
    ax1.set_ylabel("Score")
    ax1.set_title("Open-set neutral stream: include OTHER instead of dropping it", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")

    dist = opn["data"]["label_dist"]
    risk_counts = [dist.get(c, 0) for c in RISK_CODES]
    other_count = dist.get("OTHER", 0)
    names = ["OTHER", "GEOPOLITICAL", "SUPPLIER", "LABOR", "LOGISTICS", "FINANCIAL", "NATURAL_DISASTER", "PANDEMIC"]
    vals = [other_count, dist.get("GEOPOLITICAL", 0), dist.get("SUPPLIER", 0), dist.get("LABOR", 0),
            dist.get("LOGISTICS", 0), dist.get("FINANCIAL", 0), dist.get("NATURAL_DISASTER", 0), dist.get("PANDEMIC", 0)]
    cols = [GREY, NAVY, ORANGE, GREEN, "#5b8def", "#b36bff", RED, "#bbbbbb"]
    y_pos = np.arange(len(names))[::-1]
    ax2.barh(y_pos, vals, color=cols)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("# consensus labels")
    ax2.set_title(f"Gold distribution: risk prevalence {opn['data']['risk_prevalence']:.0%}\n"
                  f"({opn['data']['risk_consensus_n']} risk / {opn['data']['other_consensus_n']} OTHER)", fontsize=10)
    for yy, v in zip(y_pos, vals):
        ax2.text(v + 4, yy, str(v), va="center", fontsize=8, fontweight="bold")
    fig.suptitle("Final methodological upgrade — deployment-like open-set triage", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig11_open_set.png")); plt.close(fig)

    # ---------- FIG 12: mechanism-aware engine upgrade ----------
    cmp = mech["comparison_to_plain_open_set"]
    old = cmp["previous_two_stage"]
    new = cmp["mechanism_two_stage"]
    metrics = [
        ("Accuracy", "accuracy_8way"),
        ("Risk F1", "risk_f1"),
        ("Risk AUPRC", "risk_auprc"),
        ("Type Macro-F1", "risk_type_macro_f1_present"),
        ("Open Macro-F1", "macro_f1_present_labels"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.7), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(metrics)); w = 0.34
    old_vals = [old[k] for _, k in metrics]
    new_vals = [new[k] for _, k in metrics]
    b1 = ax1.bar(x - w/2, old_vals, w, color=GREY, label="Plain open-set TF-IDF")
    b2 = ax1.bar(x + w/2, new_vals, w, color=NAVY, label="Mechanism-aware hybrid")
    for group in (b1, b2):
        for r in group:
            ax1.text(r.get_x()+r.get_width()/2, r.get_height()+0.014, f"{r.get_height():.2f}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(x); ax1.set_xticklabels([m[0] for m in metrics], rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Score")
    ax1.set_title("Engine upgrade: same gold, same CV protocol", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")

    old_matrix = np.array(opn["real_open_set_cv"]["two_stage_detector_then_typer"]["confusion"]["matrix"])
    old_labels = opn["real_open_set_cv"]["two_stage_detector_then_typer"]["confusion"]["labels"]
    old_other = old_labels.index("OTHER")
    old_risk = [i for i, lab in enumerate(old_labels) if lab != "OTHER"]
    # Use explicit error breakdown from mechanism eval for the new model and derive old counts from confusion.
    new_counts = mech["error_breakdown"]["mechanism_two_stage"]["counts"]
    old_counts = {
        "false_positive_risk": int(old_matrix[old_other, old_risk].sum()),
        "false_negative_risk": int(old_matrix[old_risk, old_other].sum()),
        "risk_type_wrong": int(old_matrix[np.ix_(old_risk, old_risk)].sum() - old_matrix[old_risk, old_risk].sum()),
    }
    cats = ["false_positive_risk", "false_negative_risk", "risk_type_wrong"]
    labels_short = ["False + risk", "False - risk", "Wrong type"]
    y_pos = np.arange(len(cats))
    ax2.barh(y_pos + 0.17, [old_counts[c] for c in cats], height=0.32, color=GREY, label="Plain")
    ax2.barh(y_pos - 0.17, [new_counts[c] for c in cats], height=0.32, color=NAVY, label="Mechanism")
    ax2.set_yticks(y_pos); ax2.set_yticklabels(labels_short, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("# OOF errors")
    ax2.set_title("Root-cause errors reduced", fontsize=10)
    ax2.legend(fontsize=8)
    for yy, old_v, new_v in zip(y_pos, [old_counts[c] for c in cats], [new_counts[c] for c in cats]):
        ax2.text(old_v + 1, yy + 0.17, str(old_v), va="center", fontsize=8)
        ax2.text(new_v + 1, yy - 0.17, str(new_v), va="center", fontsize=8, fontweight="bold")
    fig.suptitle("Algorithmic contribution — mechanism-aware bilingual open-set engine", fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig12_mechanism_engine.png")); plt.close(fig)

    return master


if __name__ == "__main__":
    m = main()
    print("=== master metrics 집계 + 12개 figure 생성 ===")
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("\nfigures:", sorted(os.listdir(FIG_DIR)))
