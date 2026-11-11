"""Run the PancreaSignal agent (Phase 9) on synthetic cases.

    python scripts/run_agent.py --case c03_jaundice_high_ca199     # you review the draft in the terminal
    python scripts/run_agent.py --all --auto-approve                # every case in evals/agent_cases.yaml
    python scripts/run_agent.py --input my_case.json                # your own case (same JSON shape)
    python scripts/run_agent.py --all --auto-approve --api-url http://127.0.0.1:8000   # risk via the API
    python scripts/run_agent.py --all --auto-approve --no-rag       # offline: no guideline evidence

Reports -> reports/agent/<run_id>.md|.json, traces -> reports/traces/<run_id>.jsonl
Research prototype. Synthetic cases only. Not for clinical use.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from pancreasignal.agents.graph import AgentDeps, build_graph, make_rag_fn, run_case
from pancreasignal.agents.risk import HttpRisk, LocalRisk
from pancreasignal.agents.trace import Tracer
from pancreasignal.config import PROJECT_ROOT, REPORTS_DIR


def ask_reviewer(draft: dict, name: str) -> dict:
    """Show the draft in the terminal and ask a person to approve or reject it."""
    print("\n" + "=" * 70 + f"\nDRAFT FOR REVIEW: {draft['case_id']}  (decision: {draft['decision']})")
    if draft["probability"] is not None:
        print(f"  model output: {draft['probability']:.0%}")
    for f in draft["findings"]:
        print(f"  {f}")
    for e in draft["evidence"]:
        print(f"  Q: {e['question']}\n     {(e['answer'] or e['error'] or '')[:300]}")
    choice = ""
    while choice not in ("a", "r"):
        choice = input("Approve (a) or reject (r)? ").strip().lower()[:1]
    note = input("Note for the report (optional): ").strip()
    return {"decision": "approve" if choice == "a" else "reject", "reviewer": name, "note": note}


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--case", help="case_id from --cases")
    src.add_argument("--all", action="store_true", help="every case in --cases")
    src.add_argument("--input", help="a JSON file with one case")
    ap.add_argument("--cases", default="evals/agent_cases.yaml")
    ap.add_argument("--api-url", help="use the running Phase 6 API instead of the local model")
    ap.add_argument("--no-rag", action="store_true", help="skip guideline evidence (offline)")
    ap.add_argument("--auto-approve", action="store_true", help="approve every draft without asking")
    ap.add_argument("--reviewer", default="demo-reviewer")
    args = ap.parse_args()

    # Step 1: which cases
    if args.input:
        cases = [json.loads(Path(args.input).read_text())]
    else:
        cases = yaml.safe_load((PROJECT_ROOT / args.cases).read_text())["cases"]
        if args.case:
            cases = [c for c in cases if c["case_id"] == args.case]
            if not cases:
                raise SystemExit(f"No case {args.case!r} in {args.cases}")

    # Step 2: dependencies: risk model, RAG (optional; the run continues without it if unavailable)
    risk = HttpRisk(args.api_url) if args.api_url else LocalRisk()
    rag_fn, llm_client = None, None
    if not args.no_rag:
        try:
            rag_fn, llm_client = make_rag_fn()
        except Exception as e:  # noqa: BLE001 - no key / no index: say so and run without evidence
            print(f"[warn] RAG unavailable ({type(e).__name__}: {e}); running without guideline evidence")

    def reviewer(draft: dict) -> dict:
        if args.auto_approve:
            return {"decision": "approve", "reviewer": args.reviewer, "note": "auto-approved (demo)"}
        return ask_reviewer(draft, args.reviewer)

    # Step 3: run each case through the graph (one tracer + one thread per run)
    rows = []
    for c in cases:
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{c['case_id']}"
        tracer = Tracer(run_id, REPORTS_DIR / "traces", llm_client=llm_client)
        app = build_graph(AgentDeps(risk_client=risk, rag_fn=rag_fn, tracer=tracer,
                                    report_dir=str(REPORTS_DIR / "agent")))
        raw = {k: c[k] for k in ("case_id", "urine", "clinical") if k in c}
        final = run_case(app, raw, thread_id=run_id, reviewer=reviewer)
        p = final["risk"]["probability"] if final.get("risk") and final["critic"]["decision"] == "proceed" else None
        ok = "" if "expect" not in c else ("  OK" if (final["critic"]["decision"], sorted(f["code"] for f in final["critic"]["findings"]))
                                          == (c["expect"]["decision"], sorted(c["expect"]["findings"])) else "  MISMATCH vs expect")
        rows.append(ok.strip() != "MISMATCH vs expect")
        total_s = sum(e["duration_s"] for e in tracer.events)
        print(f"{c['case_id']:<28} {final['critic']['decision']:<8} p={'-' if p is None else f'{p:.0%}':<5} "
              f"findings={[f['code'] for f in final['critic']['findings']]} review={final['review']['decision']} "
              f"{total_s:.1f}s{ok}")
        print(f"    report -> {Path(final['report_paths']['markdown']).relative_to(PROJECT_ROOT)} | "
              f"trace -> {tracer.path.relative_to(PROJECT_ROOT)}")
    if not all(rows):
        raise SystemExit("Some cases did not match their expected critic decision.")


if __name__ == "__main__":
    main()
