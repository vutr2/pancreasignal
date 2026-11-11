# PancreaSignal

Biomarker-based risk triage for pancreatic ductal adenocarcinoma (PDAC): calibrated per-test models
(urine panel, serum miRNA) wrapped in an agentic pipeline: lab-report intake, guideline RAG,
confounder critic, human review. Every number comes from a reproducible eval harness.

> Research prototype. Not a medical device. Not for clinical use.

## Quickstart (Mac, venv + pip)

```bash
python3.14 -m venv .venv && source .venv/bin/activate   # 3.14: the version in requirements-lock.txt
pip install -e ".[dev,notebook,tracking,api,rag,agent,demo]"
python scripts/download_data.py --only urine   # needs ~/.kaggle/kaggle.json
python scripts/download_data.py --only mirna_gse59856
pytest                                         # all tests should pass (no network needed)
python -m pancreasignal.train --task pdac_vs_benign
python -m pancreasignal.train --config configs/mirna.yaml --task pdac_vs_control

# Guideline RAG: needs NVIDIA_API_KEY=... in .env (free key from build.nvidia.com)
python scripts/download_corpus.py              # 20 open-license articles (Europe PMC)
python scripts/build_rag_index.py              # chunks -> embeddings -> LanceDB
python scripts/ask_rag.py "Is CA19-9 reliable in obstructive jaundice?"
python scripts/eval_rag.py                     # 50 questions -> reports/*_rag_eval.json
python scripts/run_agent.py --all --auto-approve   # Phase 9 agent on 8 synthetic cases
python -m pancreasignal.demo                   # web demo at http://127.0.0.1:7860
python scripts/build_static_space.py           # free Hugging Face Space (runs in the browser)

python scripts/make_results.py                 # rebuild RESULTS.md + the tables below
```

## Layout

```
configs/            experiment configs (change values here, not in code); rag.yaml, rag_corpus.yaml
scripts/            data download, make_results.py (RESULTS.md + tables from reports/), RAG scripts
evals/              rag_questions.yaml (40 answerable + 10 out-of-corpus), agent_cases.yaml (8 synthetic cases)
deploy/static/      free Hugging Face Space: one HTML page + Pyodide (scripts/build_static_space.py)
deploy/space/       Docker Space for HF PRO (scripts/build_space.py)
notebooks/          01 EDA, 02 urine results, 03 serum miRNA: logic lives in src/
docs/figures/       plots used in this README
src/pancreasignal/
  data/             loading + validation: urine.py (Kaggle), geo.py (GEO miRNA)
  features.py       feature engineering
  models/           sklearn Pipelines (leak-safe), incl. elastic-net for p >> n
  evaluation/       splits (leave-one-center-out) + clinical metrics
  train.py          experiment runner -> reports/ (+ MLflow)
  reporting.py      read runs back from reports/
  explain.py        SHAP, coefficients, miRNA selection stability, batch checks
  api/              FastAPI /predict (Phase 6)
  rag/              guideline RAG (Phase 8): chunking, LanceDB index, cited answers + verifier, evaluation
  agents/           Phase 9 agent: clinical input, risk clients, critic rules, LangGraph graph, report, tracing
  demo.py           Gradio web demo (local, Docker Space); browser_demo.py: the same in Pyodide
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


### Serum miRNA (GSE59856, Phase 5)

| Model | PDAC vs control | PDAC vs biliary tract cancer | PDAC vs benign |
|---|---|---|---|
| Age + sex only (miRNA cohort) | 0.723 (0.645–0.794) | 0.498 (0.417–0.582) | 0.582 (0.439–0.725) |
| Elastic-net on serum miRNA | 1.000 (1.000–1.000) | 0.520 (0.444–0.603) | 0.999 (0.995–1.000) |

Near-perfect scores appear only where the two groups were likely collected separately; pancreatic vs biliary cancer is at chance. Read this as a batch-effect warning, not a better test ([RESULTS.md](RESULTS.md) section 8, [notebook](notebooks/03_mirna.ipynb)).


### Guideline RAG (Phase 8)

| Metric | Value |
|---|---|
| Retrieval hit@3 (hybrid) | 1.00 |
| Claims supported by cited sources | 99% |
| Refused out-of-corpus questions (10) | 100% |
| Fully correct answers (40) | 82% |

Every sentence cites its source chunks; a second model call checks each sentence against the retrieved sources, numbers are checked in code, and out-of-corpus questions are refused. Details: [RESULTS.md](RESULTS.md) section 9.
<!-- RESULTS:END -->

**Model card:** [MODEL_CARD.md](MODEL_CARD.md) (intended use, data, limitations).

## API

The urine logistic regression (trained on all 590 samples, task PDAC vs rest) behind FastAPI.

```bash
python scripts/export_model.py                 # -> models/urine_v1.joblib (+ .meta.json with its SHA-256)
uvicorn pancreasignal.api.app:app --reload     # docs at http://127.0.0.1:8000/docs
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "lyve1": {"value": 5.6, "unit": "ng/mL"}, "reg1b": {"value": 123, "unit": "ng/mL"},
  "tff1": {"value": 720, "unit": "ng/mL"}, "creatinine": {"value": 6.4, "unit": "mmol/L"},
  "age": 67, "sex": "M"}'
```

- **Units are required on every lab value** and converted to the training units (proteins ng/mL, creatinine
  mg/mL). Accepted: ng/mL, µg/L, pg/mL, ng/L, µg/mL, mg/L for proteins; mg/mL, g/L, mg/dL, mg/L, mmol/L, µmol/L
  for creatinine. Unknown units get HTTP 422; they are never guessed.
- **Response:** probability, top 3 reasons (exact per-feature contributions of the logistic regression, in
  log-odds), an out-of-distribution flag when any input is outside the training range, a unit-mistake warning
  for values 10x above anything seen in training, and standing warnings (case-control probability, no
  calibration across hospitals).
- **Safety:** the app refuses to load a model file whose SHA-256 does not match `models/urine_v1.meta.json`.
- **Docker:** `docker build -t pancreasignal-api . && docker run --rm -p 8000:8000 pancreasignal-api`
  (export the model first; it is copied into the image).

## Agent (Phase 9)

A LangGraph graph that turns one case (urine panel + optional clinical context) into a reviewed,
bilingual (English / Vietnamese) report.

```
intake -> risk (urine model) -> critic (rules) -> rag (guideline evidence) -> human_review (pause) -> report
```

```bash
python scripts/run_agent.py --case c03_jaundice_high_ca199   # you approve or reject the draft in the terminal
python scripts/run_agent.py --all --auto-approve             # all 8 synthetic cases in evals/agent_cases.yaml
python scripts/run_agent.py --all --auto-approve --api-url http://127.0.0.1:8000   # risk from the running API
```

- **Critic = fixed rules, not an LLM**, so every decision is explainable and tested. It **withholds** the estimate on a
  probable unit error, an unreadable case or a failed model call, and **warns** on out-of-distribution input, very
  dilute or concentrated urine, a raised CA19-9 during jaundice (possible false positive) and a near-zero CA19-9
  (possibly Lewis negative). Thresholds that are common practice rather than from the corpus are marked HEURISTIC in
  `src/pancreasignal/agents/critic.py`.
- **Why it matters, from the demo cases:** a LYVE1 value with the wrong unit makes the model output 100%; the critic
  blocks it. Control-like urine that is very dilute is pushed to 41% by the creatinine division; the critic flags it.
- **Human in the loop:** the run pauses (`interrupt`) with a draft; it resumes only with an approve/reject decision.
  Rejected cases are still written, as an audit trail.
- **Report:** built from templates and the cited evidence, never free text from an LLM. The Vietnamese part is built
  from the same fields, not machine-translated; guideline evidence stays verbatim in English with its citations.
- **Tracing:** every node run (duration, LLM calls, cache hits, LLM time, errors) goes to
  `reports/traces/<run_id>.jsonl`. Langfuse would need Docker; local JSONL keeps the same data without it.
- **Tests:** `tests/test_agent.py` checks every critic rule, pause/resume, reject, failures of the model and of the RAG,
  and that the 8 demo cases get their expected decision.

## Web demo (Hugging Face Space)

Two versions of the same demo (the run pauses for an approve/reject decision, then shows the bilingual
report and the per-step trace):

| | Free static Space | Server version (local, or Docker Space on HF PRO) |
|---|---|---|
| Runs | in the visitor's browser ([Pyodide](https://pyodide.org)) | Gradio app (`python -m pancreasignal.demo`) |
| Agent | the same step functions (`agents/steps.py`), run in a row | LangGraph graph: checkpointer, `interrupt()` |
| Urine model | pure Python (`api/light.py`) at once, then **numpy + pandas + scikit-learn 1.8** loaded in the background: a real Pipeline rebuilt from the coefficients (the 1.9 pickle cannot load there), checked against the pure-Python engine in the page | the exported `.joblib` |
| Guideline evidence | snapshot of the RAG's answers to the critic's fixed questions | snapshot, or the live RAG |
| Free questions to the RAG | no (would need an API key, hence a server) | yes, with `NVIDIA_API_KEY` as a Space secret, rationed |

```bash
# free static Space (Gradio/Docker Spaces need HF PRO)
python scripts/build_static_space.py           # -> dist/static_space/ (9 files, no secrets, no pickle)
python -m http.server 8000 --directory dist/static_space   # try it locally
hf upload <hf-username>/pancreasignal dist/static_space . --repo-type space

# server version
python -m pancreasignal.demo                   # local: http://127.0.0.1:7860
python scripts/build_space.py                  # -> dist/space/ (Docker Space, needs HF PRO)
python scripts/build_space.py --set-secret <hf-username>/<space>   # .env key -> Space secret (never a file)
```

- **Guideline evidence without an API key:** the critic only asks fixed questions (`critic.RAG_QUESTIONS`), so
  their cited answers are computed once by the RAG (`reports/demo/evidence_snapshot.json`).
- **No key in any uploaded file:** both build scripts copy an allow-list of files and refuse to finish if a file
  looks like an NVIDIA or Hugging Face key.
- **Tests:** `tests/test_light_model.py` (pure-Python copy and rebuilt Pipeline = the pickled model on the demo
  cases and 300 random patients; the browser code needs only what Pyodide ships), `tests/test_browser_demo.py`
  (both engines, all 8 cases), `tests/test_demo.py`.

## Data

Debernardi et al., *PLOS Medicine* 2020, urinary biomarker panel (via Kaggle). See `data/manifest.json` for exact file hashes.
