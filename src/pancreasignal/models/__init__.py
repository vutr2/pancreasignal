"""Model builders. ⚠️ YOUR CODE (Phase 2 + 4). Tests: tests/test_models.py

Every model is an sklearn Pipeline so that anything "fitted" (scaler, imputer, calibrator)
only ever sees the training fold. This is the main defense against data leakage.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def build_logreg(params: dict) -> Pipeline:
    """Pipeline: StandardScaler -> LogisticRegression(**params).
    The simplest strong baseline. Its coefficients are easy to explain in an interview.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(**params)),
        ]
    )
    


def build_xgboost(params: dict) -> Pipeline:
    """Pipeline: XGBClassifier(**params, eval_metric='logloss'). No scaler needed for trees."""

    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("xgb", XGBClassifier(**params, eval_metric="logloss")),
        ]
    )


def build_model(cfg: dict) -> Pipeline:
    """(Done.) Pick the builder from cfg['model']['name'] and wrap with calibration if asked."""
    m = cfg["model"]
    builders = {"logreg": build_logreg, "xgboost": build_xgboost}
    if m["name"] not in builders:
        raise ValueError(f"Unknown model {m['name']!r}")
    model = builders[m["name"]](m["params"][m["name"]])
    if m.get("calibrate"):
        # TODO(Phase 4): wrap `model` in sklearn.calibration.CalibratedClassifierCV
        #   (method="isotonic", cv=5) and return it.
        model = CalibratedClassifierCV(model, method="isotonic", cv=5)
        
    return model
