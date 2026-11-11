"""Web demo of the PancreaSignal agent (Gradio), for local use and the Hugging Face Space.
Tests: tests/test_demo.py

    python -m pancreasignal.demo                 # http://127.0.0.1:7860
    python -m pancreasignal.demo --live-rag      # agent evidence from the live RAG instead of the snapshot

Tab "Agent": load a synthetic case (or type values, each with its unit) -> Run. The graph runs
intake -> risk -> critic -> rag and PAUSES at human_review with a draft report; the visitor approves
or rejects it, then the graph resumes and writes the final bilingual report.

Where the agent's guideline evidence comes from: a SNAPSHOT. The critic only ever asks the fixed
critic.RAG_QUESTIONS, so their cited answers are computed once with the Phase 8 RAG
(scripts/build_space.py) and shipped as JSON: no API key, no cost and no wait per visitor, and every
visitor sees the answer that was checked before publishing.

Tab "Ask the guidelines": free questions to the live RAG. Only on when NVIDIA_API_KEY is set (a Space
secret, or .env locally) and rationed (per visit, per day, one at a time), because every question is
paid from the owner's API quota.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import threading
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
import yaml
from langgraph.types import Command

from pancreasignal.agents.critic import RAG_QUESTIONS
from pancreasignal.agents.graph import AgentDeps, build_graph, make_rag_fn
from pancreasignal.agents.report import STATUS, build_report, headline
from pancreasignal.agents.risk import LocalRisk
from pancreasignal.agents.steps import snapshot_rag_fn
from pancreasignal.agents.trace import Tracer
from pancreasignal.config import PROJECT_ROOT
from pancreasignal.webform import (
    BILIRUBIN_UNIT_CHOICES,
    CA199_UNIT_CHOICES,
    CREATININE_UNIT_CHOICES,
    PROTEIN_UNIT_CHOICES,
    case_from_form,
    form_from_case,
)

MODEL_PATH = Path(os.environ.get("PANCREASIGNAL_MODEL", PROJECT_ROOT / "models" / "urine_v1.joblib"))
EVIDENCE_PATH = Path(os.environ.get("PANCREASIGNAL_EVIDENCE",
                                    PROJECT_ROOT / "reports" / "demo" / "evidence_snapshot.json"))
CASES_PATH = PROJECT_ROOT / "evals" / "agent_cases.yaml"
# the README results block (made by scripts/make_results.py); the Space gets a copy in docs/
RESULTS_PATHS = (PROJECT_ROOT / "docs" / "results_summary.md", PROJECT_ROOT / "README.md")
REPO_URL = os.environ.get("PANCREASIGNAL_REPO_URL", "https://github.com/vutr2/pancreasignal")
WORK_DIR = Path(tempfile.gettempdir()) / "pancreasignal_demo"   # reports + traces (not kept)
MAX_OPEN_RUNS = 200            # paused runs kept in memory; the oldest is dropped beyond this

# live RAG rationing (Space "Variables" can change them without touching code)
RAG_SESSION_LIMIT = int(os.environ.get("RAG_SESSION_LIMIT", "5"))    # questions per visit
RAG_DAILY_LIMIT = int(os.environ.get("RAG_DAILY_LIMIT", "100"))      # questions per day, all visitors
MAX_QUESTION_CHARS = 300
ASK_EXAMPLES = [*RAG_QUESTIONS,
                "How does CA19-9 compare with CEA for diagnosing pancreatic cancer in meta-analysis?",
                "What HbA1c target is recommended for adults with type 2 diabetes?"]   # out of corpus -> refusal

# ---------------------------------------------------------------- evidence for the agent
def load_snapshot(path: str | Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------- running the graph for visitors
class DemoAgent:
    """Runs the Phase 9 graph for web visitors: start() runs until the human-review pause, finish()
    resumes it with the visitor's decision. Paused runs wait in memory (one small graph + its
    checkpointer each); past `max_open_runs` the oldest is dropped."""

    def __init__(self, risk_client, rag_fn, llm_client=None, work_dir: str | Path = WORK_DIR,
                 max_open_runs: int = MAX_OPEN_RUNS):
        self.risk_client, self.rag_fn, self.llm_client = risk_client, rag_fn, llm_client
        self.work_dir, self.max_open_runs = Path(work_dir), max_open_runs
        self._runs: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()      # several visitors can run at the same time

    def start(self, raw_case: dict) -> dict:
        # Step 1: one graph + tracer per run; the run_id names the report and trace files
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw_case.get("case_id", "case")))[:40]
        run_id = f"{safe_id}_{uuid.uuid4().hex[:8]}"
        tracer = Tracer(run_id, self.work_dir / "traces", llm_client=self.llm_client)
        app = build_graph(AgentDeps(risk_client=self.risk_client, rag_fn=self.rag_fn, tracer=tracer,
                                    report_dir=str(self.work_dir / "reports")))

        # Step 2: run until the graph pauses at human_review
        config = {"configurable": {"thread_id": run_id}}
        app.invoke({"raw": raw_case}, config)
        state = app.get_state(config).values

        # Step 3: keep the paused run for finish()
        with self._lock:
            self._runs[run_id] = (app, tracer)
            while len(self._runs) > self.max_open_runs:
                self._runs.popitem(last=False)
        return {"run_id": run_id, "state": state, "trace": list(tracer.events)}

    def finish(self, run_id: str, decision: str, reviewer: str = "", note: str = "") -> dict:
        """Resume a paused run. KeyError if it is unknown or was dropped."""
        with self._lock:
            app, tracer = self._runs.pop(run_id)
        reply = {"decision": decision, "reviewer": (reviewer or "").strip() or "anonymous",
                 "note": (note or "").strip()}
        final = app.invoke(Command(resume=reply), {"configurable": {"thread_id": run_id}})
        return {"run_id": run_id, "state": final, "trace": list(tracer.events)}


def trace_rows(events: list[dict]) -> list[list]:
    return [[e["node"], e["status"], round(e["duration_s"], 3), e["llm_api_calls"], e["llm_cache_hits"]]
            for e in events]


def results_markdown(paths=RESULTS_PATHS, repo_url: str = REPO_URL) -> tuple[str, list[Path]]:
    """The README results block for the web page: figures returned separately (shown with gr.Image),
    relative links pointed at the GitHub repo."""
    for path in paths:
        if not Path(path).exists():
            continue
        found = re.search(r"<!-- RESULTS:START -->(.*?)<!-- RESULTS:END -->", Path(path).read_text(), re.DOTALL)
        if not found:
            continue
        block = found.group(1)
        images = [PROJECT_ROOT / src for src in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", block)]
        block = re.sub(r"^!\[[^\]]*\]\([^)]+\)[ \t]*$", "", block, flags=re.MULTILINE)
        block = re.sub(r"\]\((?!https?://|#)([^)]+)\)", lambda m: f"]({repo_url}/blob/main/{m.group(1)})", block)
        return re.sub(r"\n{3,}", "\n\n", block).strip(), [p for p in images if p.exists()]
    return "_Results summary not found._", []


# ---------------------------------------------------------------- live RAG ("Ask the guidelines")
class DailyBudget:
    """Questions per UTC day for the whole demo. Kept in memory, so a restart of the Space resets it."""

    def __init__(self, per_day: int):
        self.per_day, self.day, self.used = per_day, None, 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        today = datetime.now(UTC).date()
        with self._lock:
            if today != self.day:
                self.day, self.used = today, 0
            if self.used >= self.per_day:
                return False
            self.used += 1
            return True


def answer_markdown(res: dict) -> str:
    """A RAG answer with its sources linked to the article in PubMed Central."""
    lines = [res["answer"], ""]
    if res.get("refused"):
        lines.append(f"_Refused: {res.get('refusal_reason') or 'not supported by the corpus'}._")
    if res.get("citations"):
        lines += ["", "**Sources**", *[
            f"- `{c['chunk_id']}` [{c['title']}](https://pmc.ncbi.nlm.nih.gov/articles/{c['pmcid']}/) "
            f"({c['journal']}, {c['year']}), {c['section'][:90]}" for c in res["citations"]]]
    if res.get("model_time_s") is not None:
        lines += ["", f"_Model time: {res['model_time_s']:.1f} s_"]
    return "\n".join(lines)


# ---------------------------------------------------------------- page
INTRO = f"""# PancreaSignal
Urine-biomarker triage for pancreatic cancer (PDAC), inside an agent with a rule-based critic, cited
guideline evidence and a human reviewer. [Code, evaluation and limitations on GitHub]({REPO_URL}).

> **Research prototype. Not a medical device. Synthetic cases only: do not enter real patient data.**
"""

HOW_IT_WORKS = """### Pipeline (LangGraph)

`intake -> risk -> critic -> rag -> human_review (pause) -> report`

- **intake** validates the case. Every lab value needs a unit; unknown units are rejected, never guessed.
- **risk** is a logistic regression on urine LYVE1, REG1B and TFF1 (each divided by urine creatinine),
  age and sex, trained on {n_train} patients from 4 hospitals (Debernardi et al., *PLOS Medicine* 2020,
  CC BY 4.0). Its output is a probability in case-control data, not a screening risk.
- **critic** is a set of fixed rules, not an LLM, so every decision is explainable and tested. It withholds
  the estimate on a probable unit error, an unreadable case or a failed model call, and warns on
  out-of-range input, very dilute or concentrated urine, a raised CA19-9 during jaundice and a near-zero
  CA19-9.
- **rag** turns each warning into a question for the guideline RAG (20 open-license articles, hybrid
  search, every sentence cited and checked by a second model call). {evidence_note}
- **human_review** pauses the run until a person approves or rejects it. Rejected reports are still
  written, as an audit trail.
- **report** is English + Vietnamese, built from templates and the cited evidence, never free text from
  an LLM.

Every node run is traced (the table next to the report: duration, LLM calls, cache hits).
"""

PAUSED = ("**Paused for human review.** The graph stopped at `human_review`. "
          "Read the draft below, then approve or reject it.\n\n")


def build_ui(agent: DemoAgent, cases: list[dict], evidence_note: str, ask_fn=None, ask_off_reason: str = "",
             session_limit: int = RAG_SESSION_LIMIT, daily_limit: int = RAG_DAILY_LIMIT) -> gr.Blocks:
    by_id = {c["case_id"]: c for c in cases}
    shown = by_id.get("c03_jaundice_high_ca199") or (cases[0] if cases else {"case_id": "my_case"})
    first = form_from_case(shown)     # the case filled in when the page opens
    budget = DailyBudget(daily_limit)
    n_train = getattr(agent.risk_client, "bundle", {}).get("meta", {}).get("n_train", "?")

    # Step 1: event handlers
    def on_example(case_id):
        return form_from_case(by_id[case_id]) if case_id in by_id else [gr.update()] * len(form)

    def on_run(*values):
        res = agent.start(case_from_form(*values))
        return (res["run_id"], PAUSED + headline(res["state"]), trace_rows(res["trace"]),
                build_report(res["state"])["markdown"], gr.update(interactive=True), gr.update(interactive=True),
                gr.update(value=None, visible=False))

    def on_review(decision):
        def handler(run_id, reviewer, note):
            if not run_id:
                raise gr.Error("Run the agent first.")
            try:
                res = agent.finish(run_id, decision, reviewer, note)
            except KeyError:
                raise gr.Error("This draft has expired. Run the agent again.") from None
            state, md_path = res["state"], res["state"]["report_paths"]["markdown"]
            status = (f"**Final report: {STATUS[decision][0]}** ({state['review']['reviewer']}).\n\n"
                      + headline(state))
            return (None, status, trace_rows(res["trace"]), Path(md_path).read_text(),
                    gr.update(interactive=False), gr.update(interactive=False), gr.update(value=md_path, visible=True))
        return handler

    def on_ask(question, used):
        question = (question or "").strip()
        if not question:
            raise gr.Error("Type a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise gr.Error(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
        if used >= session_limit:
            raise gr.Error(f"This demo allows {session_limit} questions per visit. To ask more, run it locally (see GitHub).")
        if not budget.take():
            raise gr.Error("Today's question budget for this demo is used up. Try again tomorrow, or run it locally.")
        return answer_markdown(ask_fn(question)), used + 1, f"Questions left in this visit: {session_limit - used - 1}"

    # Step 2: layout
    with gr.Blocks(title="PancreaSignal demo") as ui:
        gr.Markdown(INTRO)
        run_id = gr.State(None)
        with gr.Tab("Agent"), gr.Row():
            with gr.Column(scale=2, min_width=320):
                example = gr.Dropdown([(f"{c['case_id']}: {c.get('note', '')}", c["case_id"]) for c in cases],
                                      value=shown["case_id"] if shown["case_id"] in by_id else None,
                                      label="Load a synthetic case (or edit the values below)")
                case_id = gr.Textbox(first[0], label="Case ID")
                gr.Markdown("**Urine panel** (every value needs its unit)")
                fields = []
                for i, (name, choices) in enumerate([("LYVE1", PROTEIN_UNIT_CHOICES), ("REG1B", PROTEIN_UNIT_CHOICES),
                                                      ("TFF1", PROTEIN_UNIT_CHOICES),
                                                      ("Creatinine", CREATININE_UNIT_CHOICES)]):
                    with gr.Row():
                        fields += [gr.Number(first[1 + 2 * i], label=name, minimum=0, scale=2),
                                   gr.Dropdown(choices, value=first[2 + 2 * i], label="unit",
                                               allow_custom_value=True, scale=1)]
                with gr.Row():
                    age = gr.Number(first[9], label="Age", minimum=0, maximum=119)
                    sex = gr.Radio(["M", "F"], value=first[10], label="Sex")
                gr.Markdown("**Clinical context** (optional; the model does not use it, the critic does)")
                with gr.Row():
                    ca199 = gr.Number(first[11], label="Plasma CA19-9", minimum=0, scale=2)
                    ca199_u = gr.Dropdown(CA199_UNIT_CHOICES, value=first[12], label="unit",
                                          allow_custom_value=True, scale=1)
                with gr.Row():
                    bili = gr.Number(first[13], label="Total bilirubin", minimum=0, scale=2)
                    bili_u = gr.Dropdown(BILIRUBIN_UNIT_CHOICES, value=first[14], label="unit",
                                         allow_custom_value=True, scale=1)
                with gr.Row():
                    jaundice = gr.Checkbox(first[15], label="Jaundice")
                    nod = gr.Checkbox(first[16], label="New-onset diabetes")
                run_btn = gr.Button("1. Run agent", variant="primary")
            with gr.Column(scale=3, min_width=320):
                status = gr.Markdown("Load a case, then press **Run agent**.")
                with gr.Group():
                    reviewer = gr.Textbox("demo-reviewer", label="Reviewer")
                    note = gr.Textbox(label="Note for the report (optional)")
                    with gr.Row():
                        approve = gr.Button("2. Approve", interactive=False)
                        reject = gr.Button("2. Reject", variant="stop", interactive=False)
                download = gr.File(label="Final report (.md)", visible=False)
                report = gr.Markdown()
                trace = gr.Dataframe(headers=["node", "status", "seconds", "LLM calls", "cache hits"],
                                     label="Trace (one row per node run)", interactive=False)
        with gr.Tab("Ask the guidelines"):
            gr.Markdown("Questions go to the live guideline RAG: 20 open-license articles, every sentence cited "
                        "and checked against its source; questions outside the corpus are refused. "
                        f"Your question is sent to NVIDIA's API, so do not type personal data. "
                        f"Limits: {session_limit} questions per visit, {daily_limit} per day for everyone.")
            if ask_fn is None:
                gr.Markdown(f"**Live questions are off on this deployment** ({ask_off_reason}). "
                            "The Agent tab still shows cited evidence from the snapshot.")
            question = gr.Textbox(label="Question", max_lines=3, interactive=ask_fn is not None,
                                  placeholder=ASK_EXAMPLES[0])
            ask_btn = gr.Button("Ask", variant="primary", interactive=ask_fn is not None)
            if ask_fn is not None:
                gr.Examples([[q] for q in ASK_EXAMPLES], inputs=question)
            left = gr.Markdown()
            answer = gr.Markdown()
            used = gr.State(0)
        with gr.Tab("How it works"):
            gr.Markdown(HOW_IT_WORKS.format(n_train=n_train, evidence_note=evidence_note))
        with gr.Tab("Results and limits"):
            text, figures = results_markdown()
            gr.Markdown(text)
            for fig in figures:
                gr.Image(str(fig), show_label=False, interactive=False)

        # Step 3: wiring
        form = [case_id, *fields, age, sex, ca199, ca199_u, bili, bili_u, jaundice, nod]
        outputs = [run_id, status, trace, report, approve, reject, download]
        example.change(on_example, example, form)
        run_btn.click(on_run, form, outputs, api_name="run_agent")
        approve.click(on_review("approve"), [run_id, reviewer, note], outputs, api_name="approve")
        reject.click(on_review("reject"), [run_id, reviewer, note], outputs, api_name="reject")
        if ask_fn is not None:
            # one question at a time for the whole demo: protects the API quota and its rate limit
            ask_btn.click(on_ask, [question, used], [answer, used, left], api_name="ask", concurrency_limit=1)
    return ui


def make_demo(live_agent_evidence: bool = False) -> gr.Blocks:
    """The page with the real model, the evidence snapshot and (if a key is set) the live RAG."""
    # Step 1: the model (hash-checked) and the synthetic cases
    risk = LocalRisk(bundle_path=MODEL_PATH)
    cases = yaml.safe_load(CASES_PATH.read_text())["cases"]

    # Step 2: live RAG, only if the key and the index are there
    ask_fn, llm_client, off_reason = None, None, ""
    try:
        ask_fn, llm_client = make_rag_fn()
    except Exception as e:  # noqa: BLE001 - no key / no index / not installed: the demo runs without it
        off_reason = ("no NVIDIA_API_KEY secret" if "NVIDIA_API_KEY" in str(e) else f"{type(e).__name__}: {e}")

    # Step 3: evidence for the agent: the snapshot by default, the live RAG on request
    snapshot = load_snapshot(EVIDENCE_PATH)
    if live_agent_evidence and ask_fn is not None:
        source, rag_fn, note = "live RAG", ask_fn, "Here the answers come from the live RAG."
    elif snapshot is not None:
        source, rag_fn = "snapshot", snapshot_rag_fn(snapshot)
        note = (f"Because the critic's questions are fixed, their answers were computed once with that RAG "
                f"({snapshot['meta']['answer_model']}, {snapshot['meta']['built_at'][:10]}) and are shown "
                f"from that snapshot: no API key or cost per visitor.")
    else:
        source, rag_fn = "none", None
        note = "No evidence snapshot was found, so this deployment shows no guideline evidence."
    print(f"[demo] agent evidence: {source} | live questions: {'on' if ask_fn else f'off ({off_reason})'}")
    agent = DemoAgent(risk, rag_fn, llm_client=llm_client if source == "live RAG" else None)
    return build_ui(agent, cases, note, ask_fn=ask_fn, ask_off_reason=off_reason)


def main() -> None:
    ap = argparse.ArgumentParser(description="PancreaSignal web demo (Gradio)")
    ap.add_argument("--live-rag", action="store_true", help="agent evidence from the live RAG, not the snapshot")
    ap.add_argument("--port", type=int, default=None, help="default 7860 (or GRADIO_SERVER_PORT)")
    args = ap.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    make_demo(live_agent_evidence=args.live_rag).launch(
        server_port=args.port, theme=gr.themes.Soft(), allowed_paths=[str(WORK_DIR)], ssr_mode=False)


if __name__ == "__main__":
    main()
