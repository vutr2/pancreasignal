"""Feature engineering. ⚠️ YOUR CODE (Phase 2). Tests: tests/test_features.py

Why these steps?
- Urine concentration varies a lot between people (how much water they drank). Dividing a urinary
  biomarker by urinary creatinine is the standard way to correct for that.
- Biomarker values are heavily right-skewed (a few huge values). log1p() makes them closer to
  normal, which helps linear models (logistic regression) a lot. Tree models care less.
"""
from __future__ import annotations

import pandas as pd


def normalize_by_creatinine(df: pd.DataFrame, biomarkers: list[str]) -> pd.DataFrame:
    """Return a COPY of df where each biomarker column is divided by df['creatinine'].

    - Do not modify the input DataFrame.
    - Keep column names the same.
    - Leave non-biomarker columns untouched.
    """
    raise NotImplementedError("TODO(Phase 2): implement normalize_by_creatinine")


def log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a COPY of df with np.log1p applied to `columns`."""
    raise NotImplementedError("TODO(Phase 2): implement log_transform")


def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of df where 'sex' ('M'/'F') becomes 1/0 (int)."""
    raise NotImplementedError("TODO(Phase 2): implement encode_sex")


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
    raise NotImplementedError("TODO(Phase 2): implement build_features")
