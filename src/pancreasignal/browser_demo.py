"""Python side of the FREE static demo (Hugging Face static Space). Tests: tests/test_browser_demo.py

It runs inside the visitor's browser with Pyodide (Python compiled to WebAssembly): no server, no API
key, no cost, and nothing typed on the page leaves the browser. deploy/static/index.html loads Pyodide,
unpacks the allow-listed pancreasignal modules (scripts/build_static_space.py), then calls:

    setup(light_json, snapshot_json, cases_json) -> JSON   once: model, evidence, demo cases
    enable_sklearn() -> JSON                       after numpy/pandas/scikit-learn are loaded (background)
    set_engine("python" | "sklearn")               which copy of the model draft() uses
    draft(form_json) -> JSON                       intake -> risk -> critic -> rag, then PAUSE for review
    finish(decision, reviewer, note) -> JSON       the reviewer's decision -> final bilingual report

What runs here is the real code: the agent's step functions (agents/steps.py, also used by the LangGraph
graph), the critic rules, the report templates and unit validation. Two engines for the urine model:
  python  : api/light.py, pure Python. Needs only pydantic, so the page is usable within seconds.
  sklearn : a real scikit-learn Pipeline rebuilt from the same numbers (api/light.to_sklearn_bundle; the
            .joblib cannot be unpickled with the browser's scikit-learn 1.8), served by the normal code
            path (api/bundle.predict_row: pandas build_features + predict_proba). Needs ~27 MB more.
Both are checked against each other on the demo cases when the sklearn engine starts.
"""
from __future__ import annotations

import json

from pancreasignal.agents.report import STATUS, build_report, headline
from pancreasignal.agents.risk import LocalRisk
from pancreasignal.agents.steps import finish_review, run_until_review, snapshot_rag_fn
from pancreasignal.webform import (
    BILIRUBIN_UNIT_CHOICES,
    CA199_UNIT_CHOICES,
    CREATININE_UNIT_CHOICES,
    PROTEIN_UNIT_CHOICES,
    case_from_form,
    form_from_case,
)

ENGINE_LABELS = {"python": "pure Python", "sklearn": "numpy + scikit-learn"}
# one visitor per browser tab: the loaded model engines, the evidence and the run waiting for review
_session: dict = {"engines": {}, "engine": None, "light": None, "cases": [], "rag_fn": None, "state": None}


def setup(light_json: str, snapshot_json: str | None, cases_json: str) -> str:
    """Load the model and the evidence; return what the page needs to build its form."""
    # Step 1: the pure-Python engine (always available) and the evidence snapshot
    light = json.loads(light_json)
    cases = json.loads(cases_json)
    _session.update(engines={"python": LocalRisk(bundle=light)}, engine="python", light=light, cases=cases,
                    state=None, rag_fn=snapshot_rag_fn(json.loads(snapshot_json)) if snapshot_json else None)

    # Step 2: units for the dropdowns + every demo case already turned into form values
    meta = light["meta"]
    return json.dumps({
        "units": {"protein": PROTEIN_UNIT_CHOICES, "creatinine": CREATININE_UNIT_CHOICES,
                  "ca19_9": CA199_UNIT_CHOICES, "bilirubin": BILIRUBIN_UNIT_CHOICES},
        "cases": [{"case_id": c["case_id"], "note": c.get("note", ""), "form": form_from_case(c)} for c in cases],
        "model": {"task": meta["task"], "n_train": meta["n_train"],
                  "auroc": (meta.get("performance_reference") or {}).get("auroc")},
    })


def enable_sklearn() -> str:
    """Start the scikit-learn engine (numpy/pandas/scikit-learn must be importable) and check it gives
    the same probability as the pure-Python engine on every demo case that reaches the model."""
    import numpy
    import pandas
    import sklearn

    from pancreasignal.agents.steps import intake_step
    from pancreasignal.api.light import to_sklearn_bundle
    from pancreasignal.api.schemas import PredictRequest

    # Step 1: the real Pipeline, rebuilt from the exported numbers
    engine = LocalRisk(bundle=to_sklearn_bundle(_session["light"]))

    # Step 2: both engines on the demo cases; any disagreement is reported, never hidden
    checked, worst = 0, 0.0
    for c in _session["cases"]:
        case = intake_step({k: c[k] for k in ("case_id", "urine", "clinical") if k in c})["case"]
        if case is None:
            continue                                     # invalid on purpose (c07): never reaches a model
        request = PredictRequest.model_validate(case["urine"])
        p_py = _session["engines"]["python"].predict(request)["probability"]
        p_sk = engine.predict(request)["probability"]
        worst, checked = max(worst, abs(p_py - p_sk)), checked + 1
    _session["engines"]["sklearn"] = engine
    return json.dumps({"numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
                       "cases_checked": checked, "max_probability_difference": worst})


def set_engine(name: str) -> str:
    if name not in _session["engines"]:
        raise ValueError(f"Engine {name!r} is not loaded (loaded: {sorted(_session['engines'])})")
    _session["engine"] = name
    return ENGINE_LABELS[name]


def draft(form_json: str) -> str:
    """Run the steps before human review on the form values (a JSON list, case_from_form order)."""
    if not _session["engines"]:
        raise RuntimeError("setup() was not called")
    risk = _session["engines"][_session["engine"]]
    state, timings = run_until_review(case_from_form(*json.loads(form_json)), risk, _session["rag_fn"])
    _session["state"] = state
    return json.dumps({"case_id": state["case_id"], "decision": state["critic"]["decision"],
                       "findings": [f["code"] for f in state["critic"]["findings"]],
                       "engine": ENGINE_LABELS[_session["engine"]],
                       "headline": headline(state), "markdown": build_report(state)["markdown"],
                       "timings": timings})


def finish(decision: str, reviewer: str = "", note: str = "") -> str:
    """Apply the reviewer's decision to the paused run and return the final report."""
    if _session["state"] is None:
        raise RuntimeError("Run the agent first (or this draft was already reviewed).")
    state = finish_review(_session["state"], {"decision": decision, "reviewer": (reviewer or "").strip() or "anonymous",
                                              "note": (note or "").strip()})
    _session["state"] = None              # a draft is reviewed once
    report = build_report(state)
    return json.dumps({"case_id": state["case_id"], "status": STATUS[decision][0], "headline": headline(state),
                       "markdown": report["markdown"], "summary": report["summary"]})
