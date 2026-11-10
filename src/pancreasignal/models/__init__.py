"""Model builders. ⚠️ YOUR CODE (Phase 2 + 4). Tests: tests/test_models.py

Every model is an sklearn Pipeline so that anything "fitted" (scaler, imputer, calibrator)
only ever sees the training fold. This is the main defense against data leakage.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline


def build_logreg(params: dict) -> Pipeline:
    """Pipeline: StandardScaler -> LogisticRegression(**params).
    The simplest strong baseline. Its coefficients are easy to explain in an interview.
    """
    raise NotImplementedError("TODO(Phase 2): implement build_logreg")


def build_xgboost(params: dict) -> Pipeline:
    """Pipeline: XGBClassifier(**params, eval_metric='logloss'). No scaler needed for trees."""
    raise NotImplementedError("TODO(Phase 4): implement build_xgboost")


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
        raise NotImplementedError("TODO(Phase 4): calibration")
    return model
