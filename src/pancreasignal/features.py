"""Feature engineering. ⚠️ YOUR CODE (Phase 2). Tests: tests/test_features.py

Why these steps?
- Urine concentration varies a lot between people (how much water they drank). Dividing a urinary
  biomarker by urinary creatinine is the standard way to correct for that.
- Biomarker values are heavily right-skewed (a few huge values). log1p() makes them closer to
  normal, which helps linear models (logistic regression) a lot. Tree models care less.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_by_creatinine(df: pd.DataFrame, biomarkers: list[str]) -> pd.DataFrame:
    """Return a COPY of df where each biomarker column is divided by df['creatinine'].

    - Do not modify the input DataFrame.
    - Keep column names the same.
    - Leave non-biomarker columns untouched.
    """
    result = df.copy()
    for biomarker in biomarkers:
        result[biomarker] = result[biomarker] / result["creatinine"]
    return result


def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a COPY of df with np.log1p applied to `columns`."""
    result = df.copy()
    # Fix: transform ONLY `columns`. The old version took every numeric column, so age
    # (76 -> 4.34) and creatinine were silently log-transformed too.
    result[columns] = np.log1p(result[columns])
    return result


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of df where 'sex' ('M'/'F') becomes 1/0 (int)."""
    result = df.copy()
    result["sex"] = result["sex"].map({"M": 1, "F": 0})
    # .map() turns any value not in the dict ("m", "Male", NaN) into NaN without warning.
    # Fail loudly instead: the API in Phase 6 will receive free-form input.
    if result["sex"].isna().any():
        bad = sorted(df.loc[result["sex"].isna(), "sex"].astype(str).unique())
        raise ValueError(f"sex must be 'M' or 'F', got {bad}")
    return result


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Turn raw rows into the model matrix X, driven by cfg['features'].

    Steps (in this order):
      1. take biomarkers + demographics columns
      2. if normalize_by_creatinine: normalize biomarkers
      3. if log_transform: log1p the biomarkers
      4. encode sex
    Return only the feature columns, in a fixed order.

    IMPORTANT (data leakage): nothing here may use statistics computed from the whole dataset
    (e.g. mean/std for scaling). Anything "fitted" must go inside the sklearn Pipeline in
    models/, so it is fitted on the training fold only.
    """
    features = cfg["features"]
    biomarkers = features["biomarkers"]
    demographics = features["demographics"]

    # Fix: keep creatinine for now. Step 2 divides by it; selecting only
    # biomarkers + demographics first would raise KeyError: 'creatinine'.
    result = df[biomarkers + demographics + ["creatinine"]].copy()

    # Fix: the flags live in cfg["features"], not at the top level of cfg. The old
    # cfg.get("normalize_by_creatinine") always returned None, so both steps were skipped.
    # Use [...] (not .get) for required keys: a missing/misspelled key then raises KeyError
    # instead of silently turning the feature off.
    if features["normalize_by_creatinine"]:
        result = normalize_by_creatinine(result, biomarkers)
    if features["log_transform"]:
        # Order matters: log1p(biomarker / creatinine), not log1p(biomarker) / creatinine.
        result = log_transform(result, biomarkers)

    # Only encode sex when it is a feature. The age-only baseline (configs/urine_age_only.yaml)
    # has demographics: [age], so there is no 'sex' column to encode.
    if "sex" in demographics:
        result = encode_sex(result)

    # Drop creatinine here: fixed column order, only the model features.
    return result[biomarkers + demographics]
