"""One client for every LLM / embedding call in the RAG (Phase 8). Tests: tests/test_rag.py

Why a wrapper instead of calling `openai.OpenAI` everywhere:
  - provider-agnostic: NVIDIA today, Ollama tomorrow = change base_url + model names in configs/rag.yaml
  - DISK CACHE: the same request is never sent twice. Re-running the evaluation costs nothing and
    does not burn the free tier's rate limit; results stay identical between runs.
  - timeout + retry with backoff on 429 (rate limit), timeouts and 5xx; other errors fail at once.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from pancreasignal.config import PROJECT_ROOT

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMClient:
    def __init__(self, base_url: str, api_key_env: str, cache_dir: str | Path,
                 timeout_s: float = 90, max_retries: int = 4, sdk_client=None):
        # Step 1: the API key comes from the environment (.env in the project root), never from code
        load_dotenv(PROJECT_ROOT / ".env")
        if sdk_client is None:
            from openai import OpenAI  # imported here so tests can run with a fake client
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"{api_key_env} is not set. Put it in {PROJECT_ROOT / '.env'}")
            # max_retries=0: we retry ourselves below, so every retry is visible and counted
            sdk_client = OpenAI(base_url=base_url, api_key=key, timeout=timeout_s, max_retries=0)
        self.client = sdk_client
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.calls = {"api": 0, "cache": 0, "retries": 0}
        # seconds spent in API calls, INCLUDING the original time of calls now served from the cache,
        # so latency stays meaningful when a whole evaluation is replayed from the cache
        self.elapsed_total = 0.0

    # ------------------------------------------------------------------ cache
    def _cached(self, kind: str, request: dict, send):
        """Return the cached response for `request`, or call `send()` once and cache it."""
        # Step 1: the key is a hash of EVERYTHING that changes the answer (model, messages, params)
        key = hashlib.sha256(json.dumps({"kind": kind, **request}, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / kind / key[:2] / f"{key}.json"
        if path.exists():
            self.calls["cache"] += 1
            cached = json.loads(path.read_text())
            self.elapsed_total += cached.get("elapsed_s") or 0.0
            return cached["response"]

        # Step 2: call the API, retrying only errors that can go away by themselves
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = send()
                break
            except Exception as e:  # openai errors carry .status_code; timeouts/connection errors don't
                status = getattr(e, "status_code", None)
                retryable = status in RETRYABLE_STATUS or type(e).__name__ in ("APITimeoutError", "APIConnectionError")
                if not retryable or attempt == self.max_retries:
                    raise
                self.calls["retries"] += 1
                time.sleep(2 ** (attempt + 1))  # 2, 4, 8, 16 s
        self.calls["api"] += 1
        elapsed = round(time.perf_counter() - started, 3)
        self.elapsed_total += elapsed

        # Step 3: write atomically (.part then rename) so an interrupted run never leaves a broken file
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps({"request": request, "response": response, "elapsed_s": elapsed}))
        tmp.rename(path)
        return response

    # ------------------------------------------------------------------ public API
    def chat(self, model: str, messages: list[dict], max_tokens: int = 700, temperature: float = 0.0,
             extra_body: dict | None = None, timeout_s: float | None = None,
             response_format: dict | None = None) -> str:
        """Text of the model's reply. temperature 0 = as repeatable as the provider allows.
        response_format = {"type": "json_schema", ...} makes the API GUARANTEE valid JSON of that shape
        (constrained decoding); asking for JSON in the prompt alone was ignored on short answers.
        timeout_s overrides the client timeout for this call (slow reasoning models); it is not part
        of the cache key because it does not change the answer."""
        # Normalise types BEFORE hashing: temperature 0 and 0.0 are the same request, but json.dumps
        # writes "0" vs "0.0", which gave two different cache keys (a real bug found in Phase 8)
        max_tokens, temperature = int(max_tokens), float(temperature)
        request = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "extra_body": extra_body or {}}
        if response_format:
            request["response_format"] = response_format

        def send():
            r = self.client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                    temperature=temperature, extra_body=extra_body or None,
                                                    **({"timeout": timeout_s} if timeout_s else {}),
                                                    **({"response_format": response_format} if response_format else {}))
            choice = r.choices[0]
            # A reasoning model can spend every token "thinking" and return no content (seen in Step 0)
            if choice.message.content is None:
                raise RuntimeError(f"{model} returned no content (finish_reason={choice.finish_reason}); "
                                   "raise max_tokens or turn thinking off")
            return choice.message.content

        return self._cached("chat", request, send)

    def embed(self, model: str, texts: list[str], input_type: str | None = None,
              batch_size: int = 32, extra_body: dict | None = None) -> list[list[float]]:
        """One vector per text. `input_type` = 'query' or 'passage' for asymmetric retrieval models:
        questions and documents are embedded slightly differently, which retrieves better."""
        body = {**(extra_body or {}), **({"input_type": input_type} if input_type else {})}
        vectors: list[list[float]] = []
        # Step 1: small batches, each cached on its own (an interrupted run resumes where it stopped)
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            request = {"model": model, "input": batch, "extra_body": body}

            def send(batch=batch):
                r = self.client.embeddings.create(model=model, input=batch, encoding_format="float",
                                                  extra_body=body or None)
                return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]

            vectors.extend(self._cached("embed", request, send))
        return vectors


def client_from_config(cfg: dict) -> LLMClient:
    """LLMClient built from configs/rag.yaml (the `llm` and `paths` sections)."""
    llm = cfg["llm"]
    return LLMClient(base_url=llm["base_url"], api_key_env=llm["api_key_env"],
                     cache_dir=PROJECT_ROOT / cfg["paths"]["cache_dir"], timeout_s=llm["timeout_s"])
