# Results

> Every number here is copied by script from `reports/*_metrics.json` (runs listed at the bottom).
> `reports/` is not committed, so this file is the record. Research prototype, not for clinical use.

## Setup

- **Data:** Debernardi et al. 2020 urine panel, 590 samples (file hash in `data/manifest.json`).
- **Features:** LYVE1, REG1B, TFF1 divided by urine creatinine, then `log1p`; plus age and sex.
- **Model:** logistic regression (`StandardScaler` → `LogisticRegression`, C=1.0), not calibrated.
- **Metrics:** computed on out-of-fold predictions. AUROC 95% CI by bootstrap (1000 resamples, seed 42). Sensitivity at 90% and 95% specificity.
- **Splits:**
  - `pdac_vs_benign`, `pdac_vs_rest`: leave-one-center-out with folds BPTB, ESP, LIV. UCL is skipped because it has 0 PDAC, so its 20 benign samples are not scored.
  - `pdac_vs_control`: stratified 5-fold. Leave-one-center-out is impossible here because every control is from BPTB. This is a **within-center** result and is not comparable to the other two.

## Main results (logistic regression)

| Task | Split | n (PDAC) | AUROC (95% CI) | Sens @ 90% spec | Sens @ 95% spec | Brier |
|---|---|---|---|---|---|---|
| PDAC vs control | stratified 5-fold | 382 (199) | 0.929 (0.902–0.952) | 0.809 | 0.719 | 0.107 |
| PDAC vs benign | leave-one-center-out | 387 (199) | 0.788 (0.742–0.831) | 0.437 | 0.291 | 0.189 |
| PDAC vs rest | leave-one-center-out | 570 (199) | 0.785 (0.745–0.823) | 0.538 | 0.402 | 0.187 |

## Per fold

| Task | Fold | n | AUROC |
|---|---|---|---|
| PDAC vs control | fold0 | 77 | 0.968 |
| PDAC vs control | fold1 | 77 | 0.897 |
| PDAC vs control | fold2 | 76 | 0.909 |
| PDAC vs control | fold3 | 76 | 0.952 |
| PDAC vs control | fold4 | 76 | 0.953 |
| PDAC vs benign | BPTB | 226 | 0.804 |
| PDAC vs benign | ESP | 29 | 0.819 |
| PDAC vs benign | LIV | 132 | 0.808 |
| PDAC vs rest | BPTB | 409 | 0.834 |
| PDAC vs rest | ESP | 29 | 0.819 |
| PDAC vs rest | LIV | 132 | 0.800 |

## Head-to-head vs plasma CA19-9 (same rows)

Only rows that have a CA19-9 value. Model and CA19-9 are scored on exactly the same patients.

| Task | n (PDAC) | Model AUROC (95% CI) | CA19-9 AUROC (95% CI) | Sens @ 95% spec: model | Sens @ 95% spec: CA19-9 |
|---|---|---|---|---|---|
| PDAC vs control | 242 (150) | 0.958 (0.936–0.977) | 0.963 (0.936–0.982) | 0.793 | 0.887 |
| PDAC vs benign | 258 (150) | 0.780 (0.723–0.836) | 0.890 (0.847–0.928) | 0.260 | 0.727 |
| PDAC vs rest | 350 (150) | 0.782 (0.732–0.830) | 0.923 (0.892–0.952) | 0.420 | 0.760 |

## Notes and limitations

1. **CA19-9 is better on the same rows.** For PDAC vs benign: AUROC 0.890 vs 0.780, and at 95% specificity CA19-9 catches 73% of cancers vs 26% for the urine model. CA19-9 is missing not at random (measured more often in PDAC, never at ESP/UCL), so this subset is not the full population.
2. **Age may be doing much of the work.** In the EDA, age alone ranked PDAC above the others with AUROC ≈ 0.74 (in-sample, not cross-validated). The model's 0.788 for PDAC vs benign is not far above that. An age-only baseline on the same splits is planned (Phase 4) before claiming biomarker value.
3. **Pooled vs per-fold AUROC.** For `pdac_vs_benign` the pooled AUROC (0.788) is lower than every single fold (0.804–0.819). Pooling also compares patients across centers (e.g. a PDAC case from LIV vs a benign case from BPTB). Those pairs are ranked worse, because each fold has its own model and because biomarker levels can shift between centers. The pooled number is the stricter one and is used above.
4. **Small folds.** The ESP fold has 29 samples and only 6 negatives, so its AUROC is very noisy.
5. **Sensitivity at fixed specificity has no CI yet.** At 95% specificity the threshold is set by the top ~5% of negatives (about 9 of 188 benign samples), so these numbers can move a lot with a few patients.
6. **PDAC vs control (0.929) is optimistic.** All controls are from BPTB while 116 of 199 PDAC are from LIV/ESP, so part of the signal may be center differences rather than disease. It is also case-control data, not a screening population.
7. **Calibration (ECE) not yet measured**: the model is not calibrated (Phase 4).

## Sanity checks (from Plan.md)

- PDAC vs control AUROC (0.929) > PDAC vs benign (0.788): yes (different splits, so only a rough check).
- Any AUROC > 0.97 (possible leakage)? no.
- Any fold much worse than the others? No; ESP is noisy because it is small.

## Reproduce

```bash
python -m pancreasignal.train --task pdac_vs_control --split stratified_kfold
python -m pancreasignal.train --task pdac_vs_benign
python -m pancreasignal.train --task pdac_vs_rest
```

Runs used:

- `reports/20261110-231427_pdac_vs_control_logreg_metrics.json`
- `reports/20261110-231432_pdac_vs_benign_logreg_metrics.json`
- `reports/20261110-231435_pdac_vs_rest_logreg_metrics.json`
