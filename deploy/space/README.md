---
title: PancreaSignal
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Urine-biomarker PDAC triage agent with human review
---

# PancreaSignal: web demo

Urine-biomarker triage for pancreatic cancer (PDAC) inside a LangGraph agent:
intake validation -> logistic regression on a urine panel -> rule-based critic -> cited guideline
evidence -> human review -> bilingual (English / Vietnamese) report.

Code, the full evaluation (leave-one-hospital-out AUROC with confidence intervals, calibration,
comparison with plasma CA19-9) and its limitations: https://github.com/vutr2/pancreasignal

> **Research prototype. Not a medical device. Synthetic cases only: do not enter real patient data.**

## What runs here

- **Agent tab:** the real exported model (`models/urine_v1.joblib`, SHA-256 checked at start-up) on 8
  synthetic cases or your own values. The run pauses for a human decision before the report is written.
  Guideline evidence comes from a snapshot of the RAG's answers to the critic's fixed questions
  (`reports/demo/evidence_snapshot.json`), so this tab needs no API key.
- **Ask the guidelines tab:** live questions to the guideline RAG. It is on only when the Space has the
  secret `NVIDIA_API_KEY`, and rationed: `RAG_SESSION_LIMIT` questions per visit (default 5) and
  `RAG_DAILY_LIMIT` per day for everyone (default 100), one at a time. Questions are sent to NVIDIA's API.
- Nothing is stored beyond the running container: reports and traces go to a temporary folder.

## Data and licenses

- **Model training data:** Debernardi S, et al. *A combination of urinary biomarker panel and PancRISK
  score for earlier detection of pancreatic cancer: A case–control study.* PLOS Medicine, 2020
  (S1 Table, CC BY 4.0). The model file holds coefficients, value ranges and percentiles, no patient rows.
- **Guideline corpus:** excerpts (chunks, verbatim) of the open-license articles below, used for search
  and quoted with citations. Each article keeps its own license; NC = non-commercial use only,
  ND = no derivatives. This demo is non-commercial.

<!-- CORPUS -->
