"""Model explanations (Phase 4): SHAP for XGBoost, coefficients for logistic regression.

These answer "WHAT does the model rely on?", not "HOW GOOD is it?". Performance numbers only come
from train.run (out-of-fold); nothing in this file goes on the CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pancreasignal.data.urine import load_urine, select_task
from pancreasignal.evaluation.splits import get_splits
from pancreasignal.features import build_features
from pancreasignal.models import build_model


def _task_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Same data and features as train.run: (rows, X, y)."""
    df = load_urine(cfg["data"]["path"])
    rows, y = select_task(df, cfg["task"])
    return rows, build_features(rows, cfg), y


def oof_shap_values(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values for every scored patient, each computed by the fold model that did NOT train
    on that patient (same splits as train.run). Returns (X, shap_values) with the same shape.

    SHAP value = how much one feature pushed this patient's prediction up (+) or down (-),
    in log-odds, compared with the average patient.
    Why out-of-fold: explaining a model on its own training rows shows what it memorised;
    explaining it on a hospital it never saw shows what it really relies on there.
    """
    import shap  # slow to import, only needed here

    # Step 1: check the config. TreeExplainer needs the raw XGBoost model, not a calibrated wrapper
    if cfg["model"]["name"] != "xgboost" or cfg["model"].get("calibrate"):
        raise ValueError("oof_shap_values needs model.name = 'xgboost' and calibrate = false")

    # Step 2: same data, features and splits as train.run
    rows, X, y = _task_data(cfg)
    splits = get_splits(rows, y, cfg)

    # Step 3: for each fold, fit on the training rows and explain the held-out rows
    parts = []
    for _, train_idx, test_idx in splits:
        model = build_model(cfg)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        explainer = shap.TreeExplainer(model.named_steps["xgb"])
        values = explainer.shap_values(X.iloc[test_idx])
        parts.append(pd.DataFrame(values, columns=X.columns, index=X.index[test_idx]))

    # Step 4: stack the folds into one table (rows of skipped centers, e.g. UCL, are absent)
    shap_df = pd.concat(parts).sort_index()
    return X.loc[shap_df.index], shap_df


def logreg_coefficients(cfg: dict) -> pd.DataFrame:
    """Fit logistic regression once on ALL rows of the task and return its coefficients,
    strongest first.

    The scaler is inside the Pipeline, so each coefficient = change in log-odds of PDAC for
    +1 standard deviation of that feature, and odds_ratio = exp(coef).
    Fitting on all rows is fine HERE because this is interpretation, not evaluation.
    """
    # Step 1: check the config
    if cfg["model"]["name"] != "logreg" or cfg["model"].get("calibrate"):
        raise ValueError("logreg_coefficients needs model.name = 'logreg' and calibrate = false")

    # Step 2: same data and features as train.run, fitted once on everything
    _, X, y = _task_data(cfg)
    model = build_model(cfg).fit(X, y)
    coef = model.named_steps["logreg"].coef_[0]

    # Step 3: one row per feature, sorted by absolute size (sign = direction of the effect)
    table = pd.DataFrame(
        {"coef_per_sd": coef, "odds_ratio_per_sd": np.exp(coef)}, index=X.columns
    )
    return table.loc[table["coef_per_sd"].abs().sort_values(ascending=False).index]
