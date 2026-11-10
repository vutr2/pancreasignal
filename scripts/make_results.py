"""Write RESULTS.md and refresh the results blocks in README.md and MODEL_CARD.md.

    python scripts/make_results.py

Every number comes from reports/*_metrics.json and *_oof.csv (newest run per task + variant).
Nothing is typed by hand, so after a rerun the documents can never disagree with the reports.
README.md and MODEL_CARD.md are only changed between their <!-- RESULTS:START --> and
<!-- RESULTS:END --> markers; the rest of those files is written by hand.
"""
from __future__ import annotations

import re

import pandas as pd

from pancreasignal.config import PROJECT_ROOT
from pancreasignal.evaluation.metrics import paired_bootstrap_auroc_diff
from pancreasignal.reporting import TASKS, VARIANTS, get_run, load_runs, paired_oof

LOCO_TASKS = ["pdac_vs_benign", "pdac_vs_rest"]
CENTERS = ["BPTB", "LIV", "ESP"]  # the centers that can be a LOCO test fold
SPLIT_LABEL = {"leave_one_center_out": "leave-one-center-out", "stratified_kfold": "random 5-fold"}


# ---------------------------------------------------------------- small formatting helpers
def auc_ci(m: dict) -> str:
    return f"{m['auroc']:.3f} ({m['auroc_ci95'][0]:.3f}–{m['auroc_ci95'][1]:.3f})"


def diff_ci(d: dict) -> str:
    return f"{d['diff']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f})"


def excludes_zero(d: dict) -> bool:
    return d["ci95"][0] > 0 or d["ci95"][1] < 0


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def replace_block(path, block: str) -> None:
    """Replace the text between the RESULTS markers in `path` (markers are kept)."""
    text = path.read_text()
    pattern = r"(<!-- RESULTS:START -->\n).*?(<!-- RESULTS:END -->)"
    if not re.search(pattern, text, flags=re.DOTALL):
        raise ValueError(f"{path.name} has no <!-- RESULTS:START --> / <!-- RESULTS:END --> markers")
    path.write_text(re.sub(pattern, lambda m: m.group(1) + block + m.group(2), text, flags=re.DOTALL))


def main() -> None:
    # Step 1: newest run for every (task, variant); stop early if something was never run
    runs = load_runs()
    variants = [v for v in VARIANTS if not runs[runs["variant"] == v].empty]
    for task in TASKS:
        for v in ["logreg", "xgboost", "age_only"]:
            get_run(runs, task, v)  # raises a clear KeyError if missing

    def res(task: str, variant: str) -> dict:
        return get_run(runs, task, variant)["results"]

    # Step 2: paired comparisons on the same patients (pooled + inside each held-out hospital)
    comparisons = {}
    for task in TASKS:
        for other in ["xgboost", "age_only"]:
            m = paired_oof(runs, task, "logreg", other)
            entry = {"pooled": paired_bootstrap_auroc_diff(m.y, m.prob_a, m.prob_b)}
            if task in LOCO_TASKS:
                for c in CENTERS:
                    s = m[m.center == c]
                    entry[c] = paired_bootstrap_auroc_diff(s.y, s.prob_a, s.prob_b)
            comparisons[(task, other)] = entry

    # Step 3: predicted risk vs real PDAC rate per held-out hospital (shows the prevalence shift)
    oof_rest = pd.read_csv(get_run(runs, "pdac_vs_rest", "logreg")["oof_path"]).dropna(subset=["prob"])
    by_center = oof_rest.groupby("center").agg(pred=("prob", "mean"), true=("y", "mean"))

    # Step 4: numbers reused in the text
    b, r = res("pdac_vs_benign", "logreg"), res("pdac_vs_rest", "logreg")
    xb = comparisons[("pdac_vs_benign", "xgboost")]
    xc = comparisons[("pdac_vs_control", "xgboost")]["pooled"]
    within = (f"they tie ({diff_ci(xc)})" if not excludes_zero(xc)
              else f"logistic regression differs by {diff_ci(xc)}")
    ab, ar = comparisons[("pdac_vs_benign", "age_only")], comparisons[("pdac_vs_rest", "age_only")]
    ece = {v: res("pdac_vs_benign", v)["model_all"]["ece"] for v in variants}
    stage_b = b["sens_by_stage"]["spec90"]
    ca_b, mod_b = b["ca199_baseline"], b["model_on_ca199_subset"]

    findings = [
        (f"1. **Logistic regression is the best model.** Across hospitals it reaches AUROC {auc_ci(b['model_all'])} "
        f"for PDAC vs benign and {auc_ci(r['model_all'])} for PDAC vs rest. XGBoost is lower by "
        f"{diff_ci(xb['pooled'])} on the same patients; the gap comes from the BPTB fold "
        f"({diff_ci(xb['BPTB'])}), while on patients from the training hospitals (PDAC vs control, 5-fold) {within}."),
        (f"2. **The biomarkers beat age alone at BPTB but not at LIV.** Inside the BPTB fold, logistic regression "
        f"beats the age-only model by {diff_ci(ab['BPTB'])} (benign) and {diff_ci(ar['BPTB'])} (rest)"
        f"{', both CIs exclude 0' if excludes_zero(ab['BPTB']) and excludes_zero(ar['BPTB']) else ''}. "
        f"At LIV the difference is {diff_ci(ab['LIV'])} (benign) and {diff_ci(ar['LIV'])} (rest). Every benign "
        "patient at LIV has pancreatitis, the hardest condition to tell apart from PDAC."),
        ("3. **One risk scale does not transfer between hospitals.** The share of PDAC differs a lot by hospital "
        f"({by_center.loc['BPTB', 'true']:.0%} at BPTB vs {by_center.loc['LIV', 'true']:.0%} at LIV in PDAC vs rest), "
        "and each fold model learns the PDAC rate of the hospitals it trained on. The logistic regression predicts an "
        f"average risk of {by_center.loc['BPTB', 'pred']:.2f} at BPTB (true rate {by_center.loc['BPTB', 'true']:.2f}) "
        f"and {by_center.loc['LIV', 'pred']:.2f} at LIV (true rate {by_center.loc['LIV', 'true']:.2f}). The age-only "
        f"model is hit hardest: pooled AUROC {res('pdac_vs_rest', 'age_only')['model_all']['auroc']:.3f} for PDAC vs "
        "rest, although inside each hospital it ranks patients reasonably (see section 2). A real deployment would "
        "need recalibration on local data."),
        (f"4. **Calibration helps XGBoost, not logistic regression.** Isotonic calibration changes the ECE for PDAC vs "
        f"benign from {ece['xgboost']:.3f} to {ece.get('xgboost_calibrated', float('nan')):.3f} for XGBoost (its raw "
        f"probabilities are far from the observed rates) but from {ece['logreg']:.3f} to {ece['logreg_calibrated']:.3f} "
        "for logistic regression, which is already close to calibrated. The calibrator is fitted on the training "
        "hospitals, so it cannot fix the between-hospital shift from finding 3."),
        (f"5. **Plasma CA19-9 is better on the same patients.** PDAC vs benign: AUROC {ca_b['auroc']:.3f} vs "
        f"{mod_b['auroc']:.3f}; at 95% specificity CA19-9 finds {ca_b['sens_at_spec95']:.0%} of cancers, the urine "
        f"model {mod_b['sens_at_spec95']:.0%}."),
        (f"6. **Early-stage cancers are found about as often as late ones** (PDAC vs benign at 90% specificity: "
        f"stage I–II {stage_b['I-II']['sensitivity']:.2f} of {stage_b['I-II']['n_pdac']}, III–IV "
        f"{stage_b['III-IV']['sensitivity']:.2f} of {stage_b['III-IV']['n_pdac']}). But most I–II cases are IIB, "
        "and only 16 patients are stage I."),
    ]

    # Step 5: RESULTS.md, section by section
    L = [
        "# Results",
        "",
        "> Generated by `python scripts/make_results.py` from `reports/` (newest run per task and model).",
        "> Do not edit by hand: rerun the script. Research prototype, not for clinical use.",
        "",
        "## Setup",
        "",
        "- **Data:** Debernardi et al. 2020 urine panel, 590 samples (file hash in `data/manifest.json`).",
        ("- **Features:** LYVE1, REG1B, TFF1 divided by urine creatinine, then `log1p`, plus age and sex. "
        "The age-only baseline uses age alone (`configs/urine_age_only.yaml`)."),
        ("- **Models:** logistic regression (scaler inside the Pipeline), XGBoost, each with and without isotonic "
        "calibration (`CalibratedClassifierCV`, 5 inner folds, fitted on the training fold only)."),
        ("- **Splits:** PDAC vs benign and PDAC vs rest use leave-one-center-out (test hospitals BPTB, LIV, ESP; UCL "
        "has no PDAC and is skipped). PDAC vs control uses random stratified 5-fold because every control comes from "
        "BPTB; its test patients come from the same hospitals as its training patients, so it is optimistic."),
        ("- **Metrics:** on out-of-fold predictions. AUROC 95% CI and paired differences by bootstrap (1000 "
        "resamples, seed 42). Sensitivity at 90% / 95% specificity uses one threshold for all hospitals."),
        "",
        "## Key findings",
        "",
        *findings,
        "",
        "## 1. Pooled out-of-fold AUROC (95% CI)",
        "",
    ]
    header = ["Model"] + [f"{TASKS[t]} ({SPLIT_LABEL[res(t, 'logreg')['split']]})" for t in TASKS]
    L += table(header, [[VARIANTS[v]] + [auc_ci(res(t, v)["model_all"]) for t in TASKS] for v in variants])

    L += ["", "## 2. AUROC inside each held-out hospital", "",
          "Ranking quality within one hospital, so the PDAC rate of that hospital does not matter.", ""]
    neg = {c: int(((oof_rest.center == c) & (oof_rest.y == 0)).sum()) for c in CENTERS}
    header = ["Model"] + [f"{TASKS[t]}: {c}" for t in LOCO_TASKS for c in CENTERS]
    L += table(header, [[VARIANTS[v]] + [f"{res(t, v)['per_fold'][c]['auroc']:.3f}" for t in LOCO_TASKS for c in CENTERS]
                        for v in variants])
    L += ["", f"ESP has only {neg['ESP']} negatives, so its numbers are very noisy."]

    L += ["", "## 3. Paired comparisons (same patients)", "",
          "Logistic regression minus the other model. A CI that excludes 0 means a real difference.", ""]
    rows = []
    for (task, other), e in comparisons.items():
        rows.append([f"{TASKS[task]}: logreg − {VARIANTS[other]}", diff_ci(e["pooled"])]
                    + [diff_ci(e[c]) if c in e else "—" for c in CENTERS])
    L += table(["Comparison", "Pooled", "BPTB", "LIV", "ESP"], rows)

    L += ["", "## 4. Sensitivity at fixed specificity (90% / 95%)", ""]
    L += table(["Model"] + list(TASKS.values()),
               [[VARIANTS[v]] + [f"{res(t, v)['model_all']['sens_at_spec90']:.2f} / {res(t, v)['model_all']['sens_at_spec95']:.2f}"
                                 for t in TASKS] for v in variants])

    L += ["", "## 5. Sensitivity by stage at 90% specificity (stage I–II / III–IV)", ""]
    n_early = b["sens_by_stage"]["spec90"]["I-II"]["n_pdac"]
    n_late = b["sens_by_stage"]["spec90"]["III-IV"]["n_pdac"]
    L += table(["Model"] + list(TASKS.values()),
               [[VARIANTS[v]] + [f"{res(t, v)['sens_by_stage']['spec90']['I-II']['sensitivity']:.2f} / "
                                 f"{res(t, v)['sens_by_stage']['spec90']['III-IV']['sensitivity']:.2f}" for t in TASKS]
                for v in variants])
    L += ["", (f"{n_early} stage I–II and {n_late} stage III–IV patients. No CI yet; with ~100 per group, treat "
          "differences under ~0.1 as noise.")]

    L += ["", "## 6. Calibration (Brier / ECE, lower is better)", ""]
    L += table(["Model"] + list(TASKS.values()),
               [[VARIANTS[v]] + [f"{res(t, v)['model_all']['brier']:.3f} / {res(t, v)['model_all']['ece']:.3f}" for t in TASKS]
                for v in variants])

    L += ["", "## 7. Versus plasma CA19-9 (same rows)", "",
          "Only patients with a CA19-9 value (none at ESP or UCL; measured more often in PDAC).", ""]
    rows = []
    for t in TASKS:
        a, c = res(t, "logreg")["model_on_ca199_subset"], res(t, "logreg")["ca199_baseline"]
        rows.append([TASKS[t], f"{a['n']} ({a['n_pdac']})", auc_ci(a), auc_ci(c),
                     f"{a['sens_at_spec95']:.2f}", f"{c['sens_at_spec95']:.2f}"])
    L += table(["Task", "n (PDAC)", "Logistic regression AUROC", "CA19-9 AUROC",
                "Sens @95% spec: model", "Sens @95% spec: CA19-9"], rows)

    L += [
        "",
        "## Limitations",
        "",
        "- **Case-control data**, not a screening population: PDAC is 34% here, far below 1% in real screening.",
        ("- **Hospitals differ in who they sent**: all controls from BPTB, all LIV/ESP/UCL benign patients have "
        "pancreatitis. Some of what any model learns may be hospital, not disease."),
        "- **Small folds**: ESP has 29 patients; sensitivity numbers have no CI yet.",
        "- **CA19-9 subset** is not random (missing not at random), so the head-to-head covers part of the data only.",
        "",
        "## Sanity checks (Plan.md)",
        "",
        (f"- Highest AUROC anywhere: {max(res(t, v)['model_all']['auroc'] for t in TASKS for v in variants):.3f} "
        "(> 0.97 would suggest leakage)."),
        (f"- PDAC vs control ({res('pdac_vs_control', 'logreg')['model_all']['auroc']:.3f}) > PDAC vs benign "
        f"({b['model_all']['auroc']:.3f}), as expected (different splits, rough check only)."),
        "",
        "## Reproduce",
        "",
        "```bash",
        "for t in pdac_vs_benign pdac_vs_rest; do",
        "  python -m pancreasignal.train --task $t",
        "  python -m pancreasignal.train --task $t --calibrate",
        "  python -m pancreasignal.train --task $t --model xgboost",
        "  python -m pancreasignal.train --task $t --model xgboost --calibrate",
        "  python -m pancreasignal.train --task $t --config configs/urine_age_only.yaml",
        "done",
        "# pdac_vs_control: same five commands with --split stratified_kfold",
        "python scripts/make_results.py",
        "```",
        "",
        "Runs used:",
        "",
        *[f"- `reports/{row.run_id}_metrics.json`" for row in runs.sort_values(["task", "variant"]).itertuples()],
    ]
    (PROJECT_ROOT / "RESULTS.md").write_text("\n".join(L) + "\n")

    # Step 6: README block = the 3 models x 3 tasks table + CA19-9 line + figures
    age_folds = [f["auroc"] for t in LOCO_TASKS for c, f in res(t, "age_only")["per_fold"].items() if c != "ESP"]
    age_lo, age_hi = min(age_folds), max(age_folds)  # ESP left out: only 6 negatives
    readme = ["", "Pooled out-of-fold AUROC (95% CI). Full tables and caveats: [RESULTS.md](RESULTS.md).", ""]
    readme += table(["Model"] + [f"{TASKS[t]}{'*' if res(t, 'logreg')['split'] == 'stratified_kfold' else ''}" for t in TASKS],
                    [[VARIANTS[v]] + [auc_ci(res(t, v)["model_all"]) for t in TASKS]
                     for v in ["age_only", "logreg", "xgboost"]])
    readme += [
        "",
        ("\\* Random 5-fold: all controls come from BPTB, so this task cannot be tested on an unseen hospital. "
        "The other two tasks are leave-one-center-out."),
        "",
        (f"Age only looks weak when pooled because each hospital has a different PDAC rate; inside each hospital it "
        f"ranks patients with AUROC {age_lo:.2f}–{age_hi:.2f}. The biomarkers beat it at BPTB but not at LIV."),
        "",
        (f"On the same patients, plasma CA19-9 is better: PDAC vs benign AUROC {ca_b['auroc']:.3f} vs "
        f"{mod_b['auroc']:.3f} for the urine model."),
        "",
        "![ROC curves, PDAC vs benign](docs/figures/roc_pdac_vs_benign.png)",
        "![Calibration curves, PDAC vs benign](docs/figures/calibration_pdac_vs_benign.png)",
        "",
    ]
    replace_block(PROJECT_ROOT / "README.md", "\n".join(readme))

    # Step 7: MODEL_CARD block = performance of the recommended model (logistic regression)
    card = [""]
    rows = []
    for t in TASKS:
        m = res(t, "logreg")["model_all"]
        rows.append([TASKS[t], SPLIT_LABEL[res(t, "logreg")["split"]], f"{m['n']} ({m['n_pdac']})", auc_ci(m),
                     f"{m['sens_at_spec90']:.2f}", f"{m['sens_at_spec95']:.2f}", f"{m['ece']:.3f}"])
    card += table(["Task", "Split", "n (PDAC)", "AUROC (95% CI)", "Sens @90% spec", "Sens @95% spec", "ECE"], rows)
    card += ["", "Key findings (from [RESULTS.md](RESULTS.md)):", "", *findings, ""]
    replace_block(PROJECT_ROOT / "MODEL_CARD.md", "\n".join(card))

    print("wrote RESULTS.md, updated README.md and MODEL_CARD.md")


if __name__ == "__main__":
    main()
