"""Evaluate the guideline RAG (Phase 8) and save reports/<time>_rag_eval.json.

    python scripts/eval_rag.py                    # retrieval (3 modes) + answers + judge
    python scripts/eval_rag.py --retrieval-only   # no answer / judge calls at all

Every LLM and embedding call is cached (data/cache/llm), so a re-run costs nothing and gives the
same numbers. Change a prompt or a model and only the affected calls are made again.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from pancreasignal.config import REPORTS_DIR, load_config
from pancreasignal.rag.evaluate import evaluate_answers, evaluate_retrieval, load_questions
from pancreasignal.rag.index import open_index
from pancreasignal.rag.llm import client_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    # Step 1: config, questions, index, client
    cfg = load_config("configs/rag.yaml")
    questions = load_questions(cfg["paths"]["questions"])
    table, client = open_index(cfg), client_from_config(cfg)

    # Step 2: retrieval for all three modes (cheap: only query embeddings, all cached after the first run)
    report = {"config": cfg, "n_questions": len(questions),
              "retrieval": evaluate_retrieval(questions, table, client, cfg)}
    for mode, r in report["retrieval"].items():
        print(f"retrieval {mode:<6} hit@1 {r['hit@1']:.2f}  hit@3 {r['hit@3']:.2f}  hit@6 {r['hit@6']:.2f}  "
              f"MRR {r['mrr']:.2f}  missed {r['missed']}")

    # Step 3: answers + judge (the expensive part)
    if not args.retrieval_only:
        records, summary = evaluate_answers(questions, table, client, cfg)
        report["answers"], report["summary"] = records, summary
        print(json.dumps(summary, indent=2))
    report["api_calls"] = client.calls

    # Step 4: save everything (per-question records too, so failures can be read, not just counted)
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_rag_eval.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"API calls {client.calls} | saved -> reports/{out.name}")


if __name__ == "__main__":
    main()
