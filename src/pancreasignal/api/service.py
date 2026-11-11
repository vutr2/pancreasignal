"""Turn a validated request into a full prediction response (Phase 6, shared with the Phase 9 agent).

Lives outside app.py so the HTTP endpoint and the agent's in-process risk client run EXACTLY the
same code: one place for unit conversion, reasons and warnings, so the two can never disagree.
"""
from __future__ import annotations

from pancreasignal.api.bundle import predict_row
from pancreasignal.api.schemas import PredictRequest, PredictResponse

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
    # "at percentile 44" reads clearly; "higher than 44% of samples ... lowers the risk" was confusing
    if feature == "age":
        return f"Age {request.age:g} (percentile {percentile:.0f} of training patients) {direction} the risk."
    return (f"{feature} per creatinine is at percentile {percentile:.0f} of training samples; "
            f"it {direction} the risk.")


def build_prediction(bundle: dict, request: PredictRequest) -> PredictResponse:
    """Probability, top-3 reasons, out-of-distribution flag and warnings for one validated request."""
    # Step 1: convert every value to the units the model was trained on (already validated)
    inputs = request.in_dataset_units()

    # Step 2: predict + exact per-feature contributions + out-of-range check
    result = predict_row(bundle, inputs, request.age, request.sex)

    # Step 3: top 3 reasons as readable text
    reasons = []
    for c in result["contributions"][:3]:
        direction = "raises" if c["contribution_log_odds"] > 0 else "lowers"
        reasons.append({"feature": c["feature"], "direction": direction,
                        "contribution_log_odds": c["contribution_log_odds"],
                        "training_percentile": c["training_percentile"],
                        "text": _reason_text(c["feature"], direction, c["training_percentile"], request)})

    # Step 4: everything the caller needs to judge the number, not just the number
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
