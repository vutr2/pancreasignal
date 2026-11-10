# PancreaSignal

Biomarker-based risk triage for pancreatic ductal adenocarcinoma (PDAC): calibrated per-test models
(urine panel, serum miRNA) wrapped in an agentic pipeline: lab-report intake, guideline RAG,
confounder critic, human review. Every number comes from a reproducible eval harness.

> Research prototype. Not a medical device. Not for clinical use.

## Quickstart (Mac, venv + pip)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook,tracking]"
python scripts/download_data.py --only urine   # needs ~/.kaggle/kaggle.json
pytest                                         # all tests should pass
python -m pancreasignal.train --task pdac_vs_benign
python scripts/make_results.py                 # rebuild RESULTS.md + the table below
```

## Layout

```
configs/            experiment configs (change values here, not in code)
scripts/            data download, make_results.py (RESULTS.md + tables from reports/)
notebooks/          01 EDA, 02 results plots: logic lives in src/
docs/figures/       plots used in this README
src/pancreasignal/
  data/             loading + validation
  features.py       feature engineering
  models/           sklearn Pipelines (leak-safe)
  evaluation/       splits (leave-one-center-out) + clinical metrics
  train.py          experiment runner -> reports/ (+ MLflow)
  reporting.py      read runs back from reports/
  explain.py        SHAP (XGBoost), coefficients (logistic regression)
  api/  agents/     Phase 6+
tests/              unit tests = definition of done
```

## Results

<!-- RESULTS:START -->

Pooled out-of-fold AUROC (95% CI). Full tables and caveats: [RESULTS.md](RESULTS.md).

| Model | PDAC vs control* | PDAC vs benign | PDAC vs rest |
|---|---|---|---|
| Age only (logistic regression) | 0.736 (0.686–0.786) | 0.661 (0.610–0.717) | 0.512 (0.462–0.560) |
| Logistic regression | 0.929 (0.902–0.952) | 0.788 (0.742–0.831) | 0.785 (0.745–0.823) |
| XGBoost | 0.929 (0.904–0.951) | 0.747 (0.697–0.792) | 0.744 (0.704–0.782) |

\* Random 5-fold: all controls come from BPTB, so this task cannot be tested on an unseen hospital. The other two tasks are leave-one-center-out.

Age only looks weak when pooled because each hospital has a different PDAC rate; inside each hospital it ranks patients with AUROC 0.70–0.80. The biomarkers beat it at BPTB but not at LIV.

On the same patients, plasma CA19-9 is better: PDAC vs benign AUROC 0.890 vs 0.780 for the urine model.

![ROC curves, PDAC vs benign](docs/figures/roc_pdac_vs_benign.png)
![Calibration curves, PDAC vs benign](docs/figures/calibration_pdac_vs_benign.png)
<!-- RESULTS:END -->

**Model card:** [MODEL_CARD.md](MODEL_CARD.md) (intended use, data, limitations).

## Data

Debernardi et al., *PLOS Medicine* 2020, urinary biomarker panel (via Kaggle). See `data/manifest.json` for exact file hashes.
