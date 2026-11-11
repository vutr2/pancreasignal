"""Python side of the FREE static demo (Hugging Face static Space). Tests: tests/test_browser_demo.py

It runs inside the visitor's browser with Pyodide (Python compiled to WebAssembly): no server, no API
key, no cost, and nothing typed on the page leaves the browser. deploy/static/index.html loads Pyodide,
unpacks the allow-listed pancreasignal modules (scripts/build_static_space.py), then calls:

    setup(light_json, snapshot_json, cases_json) -> JSON   once: model, evidence, demo cases
    draft(form_json) -> JSON                       intake -> risk -> critic -> rag, then PAUSE for review
    finish(decision, reviewer, note) -> JSON       the reviewer's decision -> final bilingual report

What runs here is the real code: the agent's step functions (agents/steps.py, also used by the LangGraph
graph), the critic rules, the report templates, unit validation, and the pure-Python copy of the urine
model (api/light.py, tested equal to the scikit-learn one). Guideline evidence comes from the snapshot
(the critic's questions are fixed). Needs only the standard library + pydantic.
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

_session: dict = {"risk": None, "rag_fn": None, "state": None}   # one visitor per browser tab


def setup(light_json: str, snapshot_json: str | None, cases_json: str) -> str:
    """Load the model and the evidence; return what the page needs to build its form."""
    # Step 1: the model (pure-Python copy) and the evidence snapshot
    light = json.loads(light_json)
    _session.update(risk=LocalRisk(bundle=light), state=None,
                    rag_fn=snapshot_rag_fn(json.loads(snapshot_json)) if snapshot_json else None)

    # Step 2: units for the dropdowns + every demo case already turned into form values
    cases = json.loads(cases_json)
    meta = light["meta"]
    return json.dumps({
        "units": {"protein": PROTEIN_UNIT_CHOICES, "creatinine": CREATININE_UNIT_CHOICES,
                  "ca19_9": CA199_UNIT_CHOICES, "bilirubin": BILIRUBIN_UNIT_CHOICES},
        "cases": [{"case_id": c["case_id"], "note": c.get("note", ""), "form": form_from_case(c)} for c in cases],
        "model": {"task": meta["task"], "n_train": meta["n_train"],
                  "auroc": (meta.get("performance_reference") or {}).get("auroc")},
    })


def draft(form_json: str) -> str:
    """Run the steps before human review on the form values (a JSON list, case_from_form order)."""
    if _session["risk"] is None:
        raise RuntimeError("setup() was not called")
    state, timings = run_until_review(case_from_form(*json.loads(form_json)), _session["risk"], _session["rag_fn"])
    _session["state"] = state
    return json.dumps({"case_id": state["case_id"], "decision": state["critic"]["decision"],
                       "findings": [f["code"] for f in state["critic"]["findings"]],
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
