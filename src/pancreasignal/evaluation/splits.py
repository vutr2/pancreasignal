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
    # Use plain numpy arrays. A Split stores POSITIONS (0..n-1), and with numpy a mask always
    # lines up by position. With two pandas Series, y[mask] matches rows by index LABEL, which
    # silently picks the wrong rows if `centers` and `y` have different indexes.
    centers = np.asarray(centers)
    y = np.asarray(y)
    if len(centers) != len(y):
        raise ValueError(f"centers ({len(centers)}) and y ({len(y)}) have different lengths")
    if pd.isna(centers).any():
        raise ValueError("Some rows have no center, so LOCO cannot put them in any fold")

    folds = []
    # sorted() gives the same fold order on every run, so reports are comparable.
    for center in sorted(set(centers)):
        test_mask = centers == center   # rows from the held-out hospital
        train_mask = ~test_mask          # every other row: no overlap, nothing left out

        n_pos = int(y[test_mask].sum())  # y is 0/1, so the sum = number of PDAC
        n_neg = int(test_mask.sum()) - n_pos

        # Rule 1: too few PDAC in the held-out center. Sensitivity/AUROC on 2-3 cancers is noise.
        if n_pos < min_positives:
            print(f"[LOCO] skip {center}: {n_pos} PDAC in test (< {min_positives}), {n_neg} negatives")
            continue

        # Rule 2: only one class in the test set. AUROC is undefined (needs both PDAC and non-PDAC).
        if n_pos == 0 or n_neg == 0:
            print(f"[LOCO] skip {center}: test set has one class ({n_pos} PDAC, {n_neg} negatives)")
            continue

        # Rule 3: the TRAINING side also needs both classes, or the model cannot be fitted.
        # Real case: in pdac_vs_control every control is at BPTB, so holding BPTB out leaves
        # only PDAC to train on. Skipping here gives a clear message instead of an sklearn crash.
        if len(np.unique(y[train_mask])) < 2:
            print(f"[LOCO] skip {center}: the other centers have only one class to train on")
            continue

        # np.flatnonzero(mask) = positions where mask is True
        folds.append((center, np.flatnonzero(train_mask), np.flatnonzero(test_mask)))

    return folds


def get_splits(df: pd.DataFrame, y: pd.Series, cfg: dict) -> list[Split]:
    """(Done.) Dispatch on cfg['evaluation']['split']."""
    ev = cfg["evaluation"]
    if ev["split"] == "leave_one_center_out":
        return leave_one_center_out(df[cfg["data"]["center_column"]], y)
    if ev["split"] == "stratified_kfold":
        return stratified_kfold(y, random_state=ev["random_state"])
    raise ValueError(f"Unknown split {ev['split']!r}")
