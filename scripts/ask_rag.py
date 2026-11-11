"""Ask the guideline RAG one question (Phase 8).

    python scripts/ask_rag.py "Is CA19-9 reliable in obstructive jaundice?"
    python scripts/ask_rag.py --mode fts "What cyst size is a relative indication for surgery?"

Research prototype. Not for clinical use.
"""
from __future__ import annotations

import argparse

from pancreasignal.config import load_config
from pancreasignal.rag.answer import answer_question
from pancreasignal.rag.index import open_index
from pancreasignal.rag.llm import client_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["vector", "fts", "hybrid"], help="override retrieval.mode")
    args = ap.parse_args()

    cfg = load_config("configs/rag.yaml")
    client = client_from_config(cfg)
    result = answer_question(args.question, open_index(cfg), client, cfg, mode=args.mode)

    print(f"\n{result['answer']}\n")
    if result["refused"]:
        print(f"(refused: {result['refusal_reason']})")
    for c in result["citations"]:
        print(f"  [{c['chunk_id']}] {c['title']} ({c['journal']}, {c['year']}) - {c['section'][:80]}")
    for s in result["dropped_sentences"]:
        print(f"  (dropped, not supported by its sources: {s})")
    print(f"\nretrieved: {', '.join(r['chunk_id'] for r in result['retrieved'])} | model time {result['model_time_s']} s")


if __name__ == "__main__":
    main()
