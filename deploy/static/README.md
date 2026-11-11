---
title: PancreaSignal
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: other
short_description: Urine-biomarker PDAC triage agent, runs in your browser
---

# PancreaSignal: web demo (runs in your browser)

Urine-biomarker triage for pancreatic cancer (PDAC) inside an agent:
intake validation -> logistic regression on a urine panel -> rule-based critic -> cited guideline
evidence -> human review -> bilingual (English / Vietnamese) report.

Code, the full evaluation (leave-one-hospital-out AUROC with confidence intervals, calibration,
comparison with plasma CA19-9) and its limitations: https://github.com/vutr2/pancreasignal

> **Research prototype. Not a medical device. Synthetic cases only: do not enter real patient data.**

## What runs here

A static Space: no server. [Pyodide](https://pyodide.org) runs the project's own Python in the visitor's
browser (`pancreasignal.zip`: the agent's step functions, critic rules, unit validation, report templates),
so nothing typed on the page leaves the browser.

- **Model:** `model_light.json` holds the coefficients, scaling, value ranges and percentiles of the
  served logistic regression (no patient rows). Two engines: pure Python (`api/light.py`, ready within
  seconds) and **numpy + pandas + scikit-learn**, loaded in the background (about 27 MB): a real
  scikit-learn Pipeline rebuilt from the same numbers and served by the normal prediction code. The
  `.joblib` pickle is not used (written by scikit-learn 1.9; the browser has 1.8). Tests check all three
  give the same probabilities, and the page re-checks the two engines on the demo cases at start-up.
- **Guideline evidence:** the critic only asks fixed questions, so their cited answers were computed once
  with the project's RAG and ship as `evidence_snapshot.json`. Free questions to the live RAG need an API
  key, hence a server: they are only in the full version (repository).

## Data and licenses

- **Model training data:** Debernardi S, et al. *A combination of urinary biomarker panel and PancRISK
  score for earlier detection of pancreatic cancer: A case–control study.* PLOS Medicine, 2020
  (S1 Table, CC BY 4.0).
- **Guideline evidence:** cites excerpts of the open-license articles below. Each article keeps its own
  license; NC = non-commercial use only, ND = no derivatives. This demo is non-commercial.

<!-- CORPUS -->
