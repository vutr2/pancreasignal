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
pytest                                         # see what's done / TODO
python -m pancreasignal.train --config configs/urine.yaml
```

## Layout

```
configs/            experiment configs (change values here, not in code)
scripts/            data download
notebooks/          EDA only — logic lives in src/
src/pancreasignal/
  data/             loading + validation
  features.py       feature engineering
  models/           sklearn Pipelines (leak-safe)
  evaluation/       splits (leave-one-center-out) + clinical metrics
  train.py          experiment runner -> reports/ (+ MLflow)
  api/  agents/     Phase 6+
tests/              unit tests = definition of done
```

## Results

_Fill in from `reports/*_metrics.json` once real runs exist._

| Task | Model | AUROC (95% CI) | Sens @ 95% spec | vs CA19-9 (same rows) |
|---|---|---|---|---|
| PDAC vs benign | logreg | | | |

## Data

Debernardi et al., *PLOS Medicine* 2020, urinary biomarker panel (via Kaggle). See `data/manifest.json` for exact file hashes.
