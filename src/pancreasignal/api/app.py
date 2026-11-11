"""PancreaSignal API (Phase 6).

    python scripts/export_model.py                          # once: build models/urine_v1.joblib
    uvicorn pancreasignal.api.app:app --reload              # http://127.0.0.1:8000/docs

Endpoints: GET /health, GET /model, POST /predict.
Research prototype. Not a medical device. Not for clinical use.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from pancreasignal.api.bundle import load_bundle
from pancreasignal.api.schemas import PredictRequest, PredictResponse
from pancreasignal.api.service import STANDING_WARNINGS, build_prediction  # noqa: F401 (re-export)
from pancreasignal.api.units import DATASET_UNITS, allowed_units
from pancreasignal.config import PROJECT_ROOT

DEFAULT_MODEL = PROJECT_ROOT / "models" / "urine_v1.joblib"

def create_app(bundle: dict | None = None, bundle_path: str | Path | None = None) -> FastAPI:
    """Build the app. Tests pass a ready `bundle`; uvicorn uses the default (load at startup)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Step 1: load the model ONCE at startup (not on every request), with the SHA-256 check
        if app.state.bundle is None:
            path = bundle_path or os.environ.get("PANCREASIGNAL_MODEL", DEFAULT_MODEL)
            app.state.bundle = load_bundle(path)
        yield

    app = FastAPI(title="PancreaSignal API", version="1.0.0", lifespan=lifespan,
                  description="Urine biomarker PDAC model. Research prototype, not for clinical use.")
    app.state.bundle = bundle

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": app.state.bundle is not None}

    @app.get("/model")
    def model_info() -> dict:
        """What is being served: task, training data, git commit, expected performance, units."""
        return {**app.state.bundle["meta"],
                "dataset_units": DATASET_UNITS,
                "accepted_units": {a: allowed_units(a) for a in DATASET_UNITS}}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        bundle = app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Step 2: the shared service does units, prediction, reasons and warnings (also used by the agent)
        return build_prediction(bundle, request)

    return app


app = create_app()
