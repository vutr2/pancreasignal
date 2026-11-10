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

from pancreasignal.api.bundle import load_bundle, predict_row
from pancreasignal.api.schemas import PredictRequest, PredictResponse
from pancreasignal.api.units import DATASET_UNITS, allowed_units
from pancreasignal.config import PROJECT_ROOT

DEFAULT_MODEL = PROJECT_ROOT / "models" / "urine_v1.joblib"

# Always returned: what the number is and is not (Phase 4 findings)
STANDING_WARNINGS = [
    "Research prototype. Not for clinical use.",
    "The probability comes from case-control data (34% PDAC); it is not the risk in a screening population.",
    "Predicted risk did not transfer between hospitals in validation; it would need local recalibration.",
]


def _reason_text(feature: str, direction: str, percentile: float | None, request: PredictRequest) -> str:
    """One readable sentence per reason."""
    if feature == "sex":
        return f"Sex {request.sex} {direction} the risk slightly."
    if feature == "age":
        return f"Age {request.age:g} (older than {percentile:.0f}% of training patients) {direction} the risk."
    return (f"{feature} per creatinine is higher than in {percentile:.0f}% of training samples; "
            f"it {direction} the risk.")


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

        # Step 2: convert every value to the units the model was trained on (already validated)
        inputs = request.in_dataset_units()

        # Step 3: predict + exact per-feature contributions + out-of-range check
        result = predict_row(bundle, inputs, request.age, request.sex)

        # Step 4: top 3 reasons as readable text
        reasons = []
        for c in result["contributions"][:3]:
            direction = "raises" if c["contribution_log_odds"] > 0 else "lowers"
            reasons.append({"feature": c["feature"], "direction": direction,
                            "contribution_log_odds": c["contribution_log_odds"],
                            "training_percentile": c["training_percentile"],
                            "text": _reason_text(c["feature"], direction, c["training_percentile"], request)})

        meta = bundle["meta"]
        return PredictResponse(
            probability=result["probability"],
            reasons=reasons,
            out_of_distribution=bool(result["out_of_range"]),
            out_of_range=result["out_of_range"],
            inputs_in_dataset_units=inputs,
            warnings=result["warnings"] + STANDING_WARNINGS,
            model={"name": meta["name"], "task": meta["task"], "trained_at": meta["trained_at"],
                   "performance_reference": meta["performance_reference"]},
        )

    return app


app = create_app()
