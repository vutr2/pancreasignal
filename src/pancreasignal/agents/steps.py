"""The agent's steps as plain functions (Phase 9). Tests: tests/test_agent.py, tests/test_browser_demo.py

    intake_step -> risk_step -> critic_step -> rag_step -> (human review) -> report

graph.py wires these with LangGraph (CLI, API-backed runs, the Gradio demo). The free static demo runs
the SAME functions one after the other inside the browser (Pyodide), where LangGraph is not
available: run_until_review() + finish_review() below. One implementation, two ways to run it, so the
browser demo cannot drift from the real agent. This file must stay importable without LangGraph,
scikit-learn or network access.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import ValidationError

from pancreasignal.agents.clinical import CaseInput
from pancreasignal.agents.critic import review_case
from pancreasignal.api.schemas import PredictRequest


def validation_errors(e: ValidationError) -> list[str]:
    """'urine.creatinine: Field required' style messages."""
    return [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]


def intake_step(raw: dict) -> dict:
    """Validate the case JSON; a bad case is recorded (the critic will block it), never raised."""
    try:
        case = CaseInput.model_validate(raw)
    except ValidationError as e:
        return {"case": None, "clinical": {}, "intake_errors": validation_errors(e),
                "case_id": str(raw.get("case_id", "invalid"))}
    return {"case": case.model_dump(), "clinical": case.clinical.normalised(), "intake_errors": [],
            "case_id": case.case_id}


def risk_step(case: dict | None, risk_client) -> dict:
    """No valid case -> no risk call; a failing model/API -> recorded for the critic."""
    if case is None:
        return {"risk": None, "risk_error": None}
    try:
        return {"risk": risk_client.predict(PredictRequest.model_validate(case["urine"])), "risk_error": None}
    except Exception as e:  # noqa: BLE001 - any failure (HTTP, file, model) must become a 'block'
        return {"risk": None, "risk_error": f"{type(e).__name__}: {e}"}


def critic_step(state: dict) -> dict:
    return {"critic": review_case(state["intake_errors"], state.get("risk"), state.get("clinical"),
                                  risk_error=state.get("risk_error"))}


def rag_step(findings: list[dict], rag_fn: Callable[[str], dict] | None) -> dict:
    """One guideline question per finding that has one (deduplicated, in order). Errors are kept."""
    questions = list(dict.fromkeys(f["rag_question"] for f in findings if f["rag_question"]))
    evidence = []
    for q in questions:
        if rag_fn is None:
            evidence.append({"question": q, "error": "RAG not configured"})
            continue
        try:
            res = rag_fn(q)
        except Exception as e:  # noqa: BLE001 - missing key / index / network: report it, keep going
            evidence.append({"question": q, "error": f"{type(e).__name__}: {e}"})
            continue
        evidence.append({"question": q, "answer": res["answer"], "refused": res["refused"],
                         "citations": res["citations"]})
    return {"evidence": evidence}


def review_draft(state: dict) -> dict:
    """What the human reviewer sees before deciding."""
    proceed = state["critic"]["decision"] == "proceed"
    return {"case_id": state["case_id"], "decision": state["critic"]["decision"],
            "probability": (state["risk"] or {}).get("probability") if proceed else None,
            "findings": [f"[{f['severity']}] {f['message_en']}" for f in state["critic"]["findings"]],
            "evidence": [{"question": e["question"], "answer": e.get("answer"), "error": e.get("error")}
                         for e in state["evidence"]]}


def review_step(reply) -> dict:
    """The reviewer's reply -> {'review': ...}. A malformed reply is an error, never guessed."""
    if not isinstance(reply, dict) or reply.get("decision") not in ("approve", "reject"):
        raise ValueError(f"Review reply must be {{'decision': 'approve'|'reject', ...}}, got {reply!r}")
    return {"review": {"decision": reply["decision"], "reviewer": reply.get("reviewer", "unknown"),
                       "note": reply.get("note", "")}}


def snapshot_rag_fn(snapshot: dict) -> Callable[[str], dict]:
    """rag_fn that answers from a precomputed evidence snapshot instead of calling the RAG
    (scripts/build_space.py makes it for every question in critic.RAG_QUESTIONS)."""
    answers = snapshot["answers"]

    def rag_fn(question: str) -> dict:
        if question not in answers:   # a critic question added after the snapshot was built
            raise KeyError("question not in the evidence snapshot; rebuild it: python scripts/build_space.py")
        return answers[question]
    return rag_fn


# ---------------------------------------------------------------- without LangGraph (browser demo)
def run_until_review(raw: dict, risk_client, rag_fn: Callable[[str], dict] | None) -> tuple[dict, list[dict]]:
    """(state, timings): the steps before human review, in order, each timed like a trace."""
    state: dict = {"raw": raw}
    timings = []
    for name, step in [("intake", lambda s: intake_step(s["raw"])),
                       ("risk", lambda s: risk_step(s["case"], risk_client)),
                       ("critic", critic_step),
                       ("rag", lambda s: rag_step(s["critic"]["findings"], rag_fn))]:
        t0 = time.perf_counter()
        state.update(step(state))
        timings.append({"node": name, "status": "ok", "duration_s": round(time.perf_counter() - t0, 4)})
    timings.append({"node": "human_review", "status": "paused", "duration_s": 0.0})
    return state, timings


def finish_review(state: dict, reply: dict) -> dict:
    """The state after the reviewer's decision (then report.build_report(state) writes the report)."""
    return {**state, **review_step(reply)}
