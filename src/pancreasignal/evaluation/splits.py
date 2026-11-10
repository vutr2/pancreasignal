"""Train/test splitting. Tests: tests/test_splits.py

A split = (fold_name, train_indices, test_indices), indices are positions (0..n-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

Split = tuple[str, np.ndarray, np.ndarray]


def stratified_kfold(y: pd.Series, n_splits: int = 5, random_state: int = 42) -> list[Split]:
    """(Done — use as a reference for the style.) Random K-fold, keeps class ratio per fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [
        (f"fold{i}", train, test)
        for i, (train, test) in enumerate(skf.split(np.zeros(len(y)), y))
    ]


def leave_one_center_out(centers: pd.Series, y: pd.Series, min_positives: int = 5) -> list[Split]:
    """⚠️ YOUR CODE (Phase 3).

    One fold per center: test = all rows from that center, train = all other rows.

    Rules:
      - fold_name = the center name (e.g. "LIV")
      - train and test must not overlap, and together cover every row
      - SKIP a center (don't make a fold) if its test set has < min_positives PDAC cases
        or contains only one class — AUROC is undefined/meaningless there.
        Print which centers were skipped and why.
      - Return folds sorted by center name (deterministic).

    Why: random K-fold mixes patients from the same hospital into train and test, so the
    model can learn hospital-specific quirks (lab machine, sample handling). Holding out a
    whole hospital tests whether the model works somewhere it has never seen.
    """
    raise NotImplementedError("TODO(Phase 3): implement leave_one_center_out")


def get_splits(df: pd.DataFrame, y: pd.Series, cfg: dict) -> list[Split]:
    """(Done.) Dispatch on cfg['evaluation']['split']."""
    ev = cfg["evaluation"]
    if ev["split"] == "leave_one_center_out":
        return leave_one_center_out(df[cfg["data"]["center_column"]], y)
    if ev["split"] == "stratified_kfold":
        return stratified_kfold(y, random_state=ev["random_state"])
    raise ValueError(f"Unknown split {ev['split']!r}")
