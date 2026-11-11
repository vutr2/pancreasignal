"""Per-node tracing for the agent (Phase 9), written to a local JSONL file. Tests: tests/test_agent.py

Plan.md asked for Langfuse (self-hosted with Docker). Docker is not installed and was ruled out as
too heavy, so the same information goes to reports/traces/<run_id>.jsonl, one line per node run:
duration, LLM calls made / served from cache, LLM time, output keys, and errors. Swapping in
Langfuse later only means sending these same events somewhere else.
"""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path


class Tracer:
    def __init__(self, run_id: str, out_dir: str | Path, llm_client=None):
        self.run_id = run_id
        self.path = Path(out_dir) / f"{run_id}.jsonl"
        self.llm = llm_client          # optional: an LLMClient whose call counters we read
        self.events: list[dict] = []

    def _llm_snapshot(self) -> tuple[int, int, float]:
        if self.llm is None:
            return 0, 0, 0.0
        return self.llm.calls["api"], self.llm.calls["cache"], self.llm.elapsed_total

    def node(self, name: str):
        """Decorator: record one event every time the node runs (also when it pauses for review)."""
        def wrap(fn):
            @wraps(fn)
            def run(state):
                # Step 1: counters before the node
                api0, cache0, llm0 = self._llm_snapshot()
                started, t0 = datetime.now(UTC).isoformat(timespec="seconds"), time.perf_counter()
                status, out, error = "ok", None, None
                try:
                    out = fn(state)
                    return out
                except Exception as e:
                    # interrupt() for human review is raised as an exception: a pause, not a failure
                    status = "paused" if type(e).__name__ == "GraphInterrupt" else "error"
                    error = None if status == "paused" else f"{type(e).__name__}: {e}"
                    raise
                finally:
                    # Step 2: one event with the differences, then append it to the file
                    api1, cache1, llm1 = self._llm_snapshot()
                    event = {"run_id": self.run_id, "node": name, "status": status, "started_at": started,
                             "duration_s": round(time.perf_counter() - t0, 3),
                             "llm_api_calls": api1 - api0, "llm_cache_hits": cache1 - cache0,
                             "llm_time_s": round(llm1 - llm0, 3),
                             "output_keys": sorted(out) if isinstance(out, dict) else [], "error": error}
                    self.events.append(event)
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a") as f:
                        f.write(json.dumps(event) + "\n")
            return run
        return wrap
