"""How the agent gets a risk estimate (Phase 9): in-process, or from the Phase 6 API over HTTP.
Tests: tests/test_agent.py

Both return the same dict as POST /predict, because both end in api/service.build_prediction:
  LocalRisk : loads the model bundle (SHA-256 checked) and calls the service directly. Default:
              no server to start, and the demo works offline except for the RAG step.
  HttpRisk  : POSTs to a running API (uvicorn pancreasignal.api.app:app). Same contract, so the
              agent can later talk to a deployed service without changing anything else.
"""
from __future__ import annotations

from pathlib import Path

from pancreasignal.api.schemas import PredictRequest
from pancreasignal.api.service import build_prediction


class LocalRisk:
    def __init__(self, bundle_path: str | Path | None = None, bundle: dict | None = None):
        # Step 1: a ready bundle (tests) or the exported model file (load_bundle checks its hash)
        # (imports here: the browser demo passes a light JSON bundle and has no scikit-learn / FastAPI)
        if bundle is None:
            from pancreasignal.api.bundle import load_bundle
            if bundle_path is None:
                from pancreasignal.api.app import DEFAULT_MODEL
                bundle_path = DEFAULT_MODEL
            bundle = load_bundle(bundle_path)
        self.bundle = bundle
        self.name = f"local:{bundle['meta']['name']}"

    def predict(self, request: PredictRequest) -> dict:
        return build_prediction(self.bundle, request).model_dump()


class HttpRisk:
    def __init__(self, base_url: str, timeout_s: float = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.name = f"http:{self.base_url}"

    def predict(self, request: PredictRequest) -> dict:
        import httpx  # only needed for this client

        # Step 1: same JSON body a curl user would send; errors are raised, not returned as data
        r = httpx.post(f"{self.base_url}/predict", json=request.model_dump(), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()
