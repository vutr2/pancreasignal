"""Model explanations (Phase 4-5): SHAP for XGBoost, coefficients for logistic regression,
selection stability for the elastic-net miRNA model.

These answer "WHAT does the model rely on?", not "HOW GOOD is it?". Performance numbers only come
from train.run (out-of-fold); nothing in this file goes on the CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pancreasignal.data.geo import MIRNA_TASKS, load_mirna
from pancreasignal.evaluation.splits import get_splits
from pancreasignal.features import build_mirna_features
from pancreasignal.models import build_model
from pancreasignal.train import load_task_data


def _task_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Same data and features as train.run: (rows, X, y). One shared loader, so they never drift."""
    return load_task_data(cfg)


def selection_stability(cfg: dict) -> pd.DataFrame:
    """Elastic-net (Phase 5): which features does each outer-fold model keep (non-zero coefficient)?

    Returns one row per feature that was kept at least once:
      n_folds   = in how many outer folds it was kept (out of len(splits))
      mean_coef = average coefficient when kept (sign = direction: + means higher in PDAC)

    Why: with p >> n, the "selected miRNAs" change from fold to fold. A feature kept in every fold
    is a much stronger finding than one kept once. Reporting a single list from one fit hides this.
    """
    # Step 1: only the elastic-net Pipeline has the steps we read below
    if cfg["model"]["name"] != "elasticnet":
        raise ValueError("selection_stability needs model.name = 'elasticnet'")

    # Step 2: same data, features and splits as train.run
    rows, X, y = _task_data(cfg)
    splits = get_splits(rows, y, cfg)

    # Step 3: fit each fold; read the names that survived the detection filter + their coefficients
    kept = []
    for fold, train_idx, _ in splits:
        model = build_model(cfg).fit(X.iloc[train_idx], y.iloc[train_idx])
        names = model[:-1].get_feature_names_out()          # after impute + detection filter
        coef = model.named_steps["clf"].coef_[0]
        kept.append(pd.Series(coef[coef != 0], index=names[coef != 0], name=fold))

    # Step 4: one table: how often each feature was kept, and its average coefficient
    table = pd.concat(kept, axis=1)                        # NaN = not kept in that fold
    out = pd.DataFrame({"n_folds": table.notna().sum(axis=1), "mean_coef": table.mean(axis=1)})
    return out.sort_values(["n_folds", "mean_coef"], key=lambda s: s.abs(), ascending=False)


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


def predict_other_groups(cfg: dict, threshold: float = 0.5) -> pd.DataFrame:
    """miRNA (Phase 5): fit the task's model on ALL rows of the task, then score the samples from
    diagnosis groups that are NOT in the task (e.g. colon cancer for pdac_vs_control). Those samples
    were never used for training, so their scores are honest.

    Returns one row per other group: n, median predicted probability, share called "PDAC".
    Why: if a "PDAC vs healthy control" model also calls colon cancer and benign disease "PDAC",
    it has learned "not a healthy control" (or a collection batch), not pancreatic cancer.
    """
    # Step 1: only the miRNA data has extra diagnosis groups outside the task
    if cfg["data"].get("dataset") != "mirna":
        raise ValueError("predict_other_groups is for the miRNA data (cfg['data']['dataset'] = 'mirna')")

    # Step 2: fit once on every row of the task
    _, X, y = _task_data(cfg)
    model = build_model(cfg).fit(X, y)

    # Step 3: the samples outside the task, turned into features by the SAME function
    df = load_mirna(cfg["data"]["path"])
    neg, pos = MIRNA_TASKS[cfg["task"]]
    others = df[~df["diagnosis"].isin(neg + pos)].reset_index(drop=True)
    prob = model.predict_proba(build_mirna_features(others, cfg))[:, 1]

    # Step 4: one summary row per group
    scored = pd.DataFrame({"diagnosis": others["diagnosis"], "prob": prob})
    return scored.groupby("diagnosis")["prob"].agg(
        n="size", median_prob="median", share_called_pdac=lambda p: float((p > threshold).mean())
    ).sort_values("share_called_pdac", ascending=False)
