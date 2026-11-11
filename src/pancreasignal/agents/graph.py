"""The PancreaSignal agent as a LangGraph graph (Phase 9). Tests: tests/test_agent.py

    intake -> risk -> critic -> rag -> human_review (pause) -> report

  intake       : validate the case JSON (units required); errors are kept, not raised
  risk         : urine model via LocalRisk (in-process) or HttpRisk (the Phase 6 API)
  critic       : fixed rules -> findings + decision (proceed / abstain)
  rag          : one guideline question per finding that has one (Phase 8 answer_question)
  human_review : interrupt(): the run PAUSES with a draft; a person approves or rejects it
  report       : bilingual markdown + json, also for rejected runs (an audit trail)

The graph is linear on purpose: every case goes through every step, and the steps themselves
decide what to do when something upstream failed. That keeps every run comparable in the traces.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from pancreasignal.agents.clinical import CaseInput
from pancreasignal.agents.critic import review_case
from pancreasignal.agents.report import build_report, save_report
from pancreasignal.agents.trace import Tracer
from pancreasignal.api.schemas import PredictRequest


class AgentState(TypedDict, total=False):
    case_id: str
    raw: dict                 # the case JSON as received
    case: dict | None         # validated CaseInput (model_dump), None if invalid
    clinical: dict            # ClinicalContext.normalised()
    intake_errors: list[str]
    risk: dict | None
    risk_error: str | None
    critic: dict
    evidence: list[dict]
    review: dict
    report_paths: dict


@dataclass
class AgentDeps:
    risk_client: object                                  # .predict(PredictRequest) -> dict
    rag_fn: Callable[[str], dict] | None                 # question -> answer_question(...) result; None = no RAG
    tracer: Tracer
    report_dir: str


def _errors(e: ValidationError) -> list[str]:
    """'urine.creatinine: Field required' style messages."""
    return [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]


def build_graph(deps: AgentDeps):
    t = deps.tracer

    @t.node("intake")
    def intake(state: AgentState) -> dict:
        # Step 1: validate; a bad case is recorded (the critic will block it), never crashes the run
        try:
            case = CaseInput.model_validate(state["raw"])
        except ValidationError as e:
            return {"case": None, "clinical": {}, "intake_errors": _errors(e),
                    "case_id": str(state["raw"].get("case_id", "invalid"))}
        return {"case": case.model_dump(), "clinical": case.clinical.normalised(), "intake_errors": [],
                "case_id": case.case_id}

    @t.node("risk")
    def risk(state: AgentState) -> dict:
        # Step 2: no valid case -> no risk call; a failing model/API -> recorded for the critic
        if state["case"] is None:
            return {"risk": None, "risk_error": None}
        try:
            return {"risk": deps.risk_client.predict(PredictRequest.model_validate(state["case"]["urine"])),
                    "risk_error": None}
        except Exception as e:  # noqa: BLE001 - any failure (HTTP, file, model) must become a 'block'
            return {"risk": None, "risk_error": f"{type(e).__name__}: {e}"}

    @t.node("critic")
    def critic(state: AgentState) -> dict:
        return {"critic": review_case(state["intake_errors"], state.get("risk"), state.get("clinical"),
                                      risk_error=state.get("risk_error"))}

    @t.node("rag")
    def rag(state: AgentState) -> dict:
        # Step 3: one guideline question per finding that has one (deduplicated, in order)
        questions = list(dict.fromkeys(f["rag_question"] for f in state["critic"]["findings"] if f["rag_question"]))
        evidence = []
        for q in questions:
            if deps.rag_fn is None:
                evidence.append({"question": q, "error": "RAG not configured"})
                continue
            try:
                res = deps.rag_fn(q)
            except Exception as e:  # noqa: BLE001 - missing key / index / network: report it, keep going
                evidence.append({"question": q, "error": f"{type(e).__name__}: {e}"})
                continue
            evidence.append({"question": q, "answer": res["answer"], "refused": res["refused"],
                             "citations": res["citations"]})
        return {"evidence": evidence}

    @t.node("human_review")
    def human_review(state: AgentState) -> dict:
        # Step 4: PAUSE. The draft goes to a person; the run resumes with their decision.
        draft = {"case_id": state["case_id"], "decision": state["critic"]["decision"],
                 "probability": (state["risk"] or {}).get("probability") if state["critic"]["decision"] == "proceed" else None,
                 "findings": [f"[{f['severity']}] {f['message_en']}" for f in state["critic"]["findings"]],
                 "evidence": [{"question": e["question"], "answer": e.get("answer"), "error": e.get("error")}
                              for e in state["evidence"]]}
        reply = interrupt(draft)
        if not isinstance(reply, dict) or reply.get("decision") not in ("approve", "reject"):
            raise ValueError(f"Review reply must be {{'decision': 'approve'|'reject', ...}}, got {reply!r}")
        return {"review": {"decision": reply["decision"], "reviewer": reply.get("reviewer", "unknown"),
                           "note": reply.get("note", "")}}

    @t.node("report")
    def report(state: AgentState) -> dict:
        return {"report_paths": save_report(build_report(state), deps.report_dir, t.run_id)}

    # Step 5: wire the steps in a line; the checkpointer keeps the state while the run is paused
    g = StateGraph(AgentState)
    for name, fn in [("intake", intake), ("risk", risk), ("critic", critic), ("rag", rag),
                     ("human_review", human_review), ("report", report)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    for a, b in [("intake", "risk"), ("risk", "critic"), ("critic", "rag"), ("rag", "human_review"),
                 ("human_review", "report")]:
        g.add_edge(a, b)
    g.add_edge("report", END)
    return g.compile(checkpointer=InMemorySaver())


def run_case(app, raw_case: dict, thread_id: str, reviewer: Callable[[dict], dict]) -> dict:
    """Run one case to the end: start, pause at human_review, ask `reviewer(draft)`, resume.
    Returns the final state."""
    config = {"configurable": {"thread_id": thread_id}}
    out = app.invoke({"raw": raw_case}, config)
    while "__interrupt__" in out:
        draft = out["__interrupt__"][0].value
        out = app.invoke(Command(resume=reviewer(draft)), config)
    return out


def make_rag_fn(cfg_path: str = "configs/rag.yaml"):
    """(rag_fn, llm_client) using the Phase 8 index and models; the client is shared with the tracer."""
    from pancreasignal.config import load_config
    from pancreasignal.rag.answer import answer_question
    from pancreasignal.rag.index import open_index
    from pancreasignal.rag.llm import client_from_config

    cfg = load_config(cfg_path)
    client, table = client_from_config(cfg), open_index(cfg)
    return (lambda question: answer_question(question, table, client, cfg)), client
