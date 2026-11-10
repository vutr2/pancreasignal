# Model card: PancreaSignal urine panel (v1)

> Research prototype. Not a medical device. Not for clinical use.

## Model details

- **What it is:** a logistic regression that outputs a probability of pancreatic ductal adenocarcinoma (PDAC)
  from one urine sample plus age and sex.
- **Inputs:** urinary LYVE1, REG1B, TFF1 (ng/ml) and creatinine (mg/ml); age (years); sex (M/F).
- **Preprocessing:** each protein is divided by urine creatinine (corrects for how diluted the urine is), then
  `log1p`. Features are standardised inside the scikit-learn Pipeline, so the scaler only sees training data.
- **Version:** v1 (Plan Phase 4), November 2026. Config: `configs/urine.yaml`.
- **Alternatives evaluated:** XGBoost, isotonic calibration of both models, and an age-only baseline
  (see [RESULTS.md](RESULTS.md)).
- **Why logistic regression:** best AUROC on unseen hospitals, closest to calibrated, and its coefficients can be
  read directly.

## Intended use

- **In scope:** research and teaching. The project shows how to evaluate a biomarker model honestly: testing on
  hospitals the model never saw, comparing with CA19-9 on the same patients, checking calibration and
  early-stage sensitivity.
- **Out of scope:** any decision about a real patient (screening, diagnosis, triage); populations not in the data;
  replacing CA19-9 or imaging.

## Data

- Debernardi S, et al. *A combination of urinary biomarker panel and PancRISK score for earlier detection of
  pancreatic cancer: A case–control study.* PLOS Medicine, 2020. Obtained via Kaggle; file hashes in
  `data/manifest.json`.
- 590 urine samples from 4 hospitals (UK: BPTB, LIV, UCL; Spain: ESP): 183 healthy controls, 208 benign
  hepatobiliary disease, 199 PDAC.
- **Case-control design:** patients were chosen because their diagnosis was already known. PDAC is 34% of this
  data, but far below 1% in a real screening population, so positive predictive value in practice would be much
  lower than these results suggest.
- **Hospitals contributed different patients:** all healthy controls are from BPTB; every benign patient at LIV,
  ESP and UCL has pancreatitis; UCL has no PDAC.
- **Plasma CA19-9 is missing for 41%** of patients, more often in the non-cancer groups (missing not at random).

## Evaluation

- All metrics use out-of-fold predictions (each patient is scored by a model that did not train on them).
- PDAC vs benign and PDAC vs rest: **leave-one-center-out**, i.e. train on three hospitals, test on the fourth.
  PDAC vs control: random stratified 5-fold, because every control comes from one hospital; its test patients
  come from the same hospitals as its training patients, so that result is optimistic.
- Metrics: AUROC with bootstrap 95% CI; sensitivity at 90% and 95% specificity; sensitivity for stage I–II vs
  III–IV; Brier score and expected calibration error (ECE); paired bootstrap for model-vs-model differences;
  plasma CA19-9 compared on exactly the same patients.

## Performance (logistic regression)

<!-- RESULTS:START -->

| Task | Split | n (PDAC) | AUROC (95% CI) | Sens @90% spec | Sens @95% spec | ECE |
|---|---|---|---|---|---|---|
| PDAC vs control | random 5-fold | 382 (199) | 0.929 (0.902–0.952) | 0.81 | 0.72 | 0.039 |
| PDAC vs benign | leave-one-center-out | 387 (199) | 0.788 (0.742–0.831) | 0.44 | 0.29 | 0.066 |
| PDAC vs rest | leave-one-center-out | 570 (199) | 0.785 (0.745–0.823) | 0.54 | 0.40 | 0.119 |

Key findings (from [RESULTS.md](RESULTS.md)):

1. **Logistic regression is the best model.** Across hospitals it reaches AUROC 0.788 (0.742–0.831) for PDAC vs benign and 0.785 (0.745–0.823) for PDAC vs rest. XGBoost is lower by +0.041 (+0.010 to +0.071) on the same patients; the gap comes from the BPTB fold (+0.052 (+0.008 to +0.098)), while on patients from the training hospitals (PDAC vs control, 5-fold) they tie (-0.000 (-0.017 to +0.015)).
2. **The biomarkers beat age alone at BPTB but not at LIV.** Inside the BPTB fold, logistic regression beats the age-only model by +0.097 (+0.044 to +0.147) (benign) and +0.135 (+0.088 to +0.183) (rest), both CIs exclude 0. At LIV the difference is +0.009 (-0.068 to +0.085) (benign) and +0.000 (-0.081 to +0.080) (rest). Every benign patient at LIV has pancreatitis, the hardest condition to tell apart from PDAC.
3. **One risk scale does not transfer between hospitals.** The share of PDAC differs a lot by hospital (20% at BPTB vs 70% at LIV in PDAC vs rest), and each fold model learns the PDAC rate of the hospitals it trained on. The logistic regression predicts an average risk of 0.43 at BPTB (true rate 0.20) and 0.47 at LIV (true rate 0.70). The age-only model is hit hardest: pooled AUROC 0.512 for PDAC vs rest, although inside each hospital it ranks patients reasonably (see section 2). A real deployment would need recalibration on local data.
4. **Calibration helps XGBoost, not logistic regression.** Isotonic calibration changes the ECE for PDAC vs benign from 0.167 to 0.090 for XGBoost (its raw probabilities are far from the observed rates) but from 0.066 to 0.071 for logistic regression, which is already close to calibrated. The calibrator is fitted on the training hospitals, so it cannot fix the between-hospital shift from finding 3.
5. **Plasma CA19-9 is better on the same patients.** PDAC vs benign: AUROC 0.890 vs 0.780; at 95% specificity CA19-9 finds 73% of cancers, the urine model 26%.
6. **Early-stage cancers are found about as often as late ones** (PDAC vs benign at 90% specificity: stage I–II 0.47 of 102, III–IV 0.40 of 97). But most I–II cases are IIB, and only 16 patients are stage I.
<!-- RESULTS:END -->

## What the model relies on

See `notebooks/02_results.ipynb`.
- In the logistic regression, **LYVE1 is the strongest feature**, then age, REG1B and TFF1; sex adds almost nothing.
- In XGBoost (out-of-fold SHAP), **age comes first**, then LYVE1 and REG1B. Leaning more on age may be part of
  why XGBoost transfers worse to a new hospital.
- Higher LYVE1, REG1B and age push the prediction towards PDAC, as expected from the original study's choice of
  panel.

## Limitations and risks

- **Worse than plasma CA19-9** on the same patients, especially at high specificity.
- **Risk scores are hospital-specific.** Predicted risk is too high at one hospital and too low at another, because
  hospitals differ in their share of PDAC. The model would need recalibration on local data before any use.
- **The biomarkers help over age at BPTB but not at LIV**, where all benign patients have pancreatitis.
- **Small numbers:** the ESP test fold has 29 patients; only 16 patients are stage I; sensitivities have no CI yet.
- **Population:** 4 European hospitals; no information on ethnicity; no screening-population data.
- **Confounders not examined:** e.g. jaundice and diabetes, which can change protein levels and CA19-9.
- **Possible harms if misused:** false positives lead to anxiety and invasive follow-up (CT, endoscopic ultrasound,
  biopsy); false negatives give false reassurance.

## Recommendations

Use as a methods demonstration only. A clinical claim would need a prospective cohort from a real screening
population (for example new-onset diabetes or high-risk families), local recalibration, and a test of whether the
urine panel adds anything **on top of** CA19-9.
