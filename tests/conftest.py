"""Shared test fixtures.

`fake_urine` is a SMALL RANDOM TABLE with the same columns as the real dataset.
It exists only so unit tests run fast and offline. Never report numbers from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_fake_urine(n_per_center: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    centers = {"AAA": n_per_center, "BBB": n_per_center, "CCC": n_per_center, "TINY": 6}
    i = 0
    for center, n in centers.items():
        for j in range(n):
            dx = [1, 2, 3][j % 3] if center != "TINY" else (3 if j < 2 else 1)
            shift = 1.0 if dx == 3 else 0.0
            rows.append({
                "sample_id": f"S{i}",
                "patient_cohort": "Cohort1",
                "sample_origin": center,
                "age": int(rng.integers(30, 85)),
                "sex": rng.choice(["M", "F"]),
                "diagnosis": dx,
                "stage": "II" if dx == 3 else np.nan,
                "benign_sample_diagnosis": "x" if dx == 2 else np.nan,
                "plasma_CA19_9": float(rng.lognormal(2 + 2 * shift, 1)) if j % 2 == 0 else np.nan,
                "creatinine": float(rng.uniform(0.2, 2.5)),
                "LYVE1": float(rng.lognormal(0 + shift, 1)),
                "REG1B": float(rng.lognormal(3 + shift, 1)),
                "TFF1": float(rng.lognormal(5 + shift, 1)),
                "REG1A": np.nan,
            })
            i += 1
    return pd.DataFrame(rows)


@pytest.fixture
def fake_urine() -> pd.DataFrame:
    return make_fake_urine()


@pytest.fixture
def cfg() -> dict:
    return {
        "data": {"path": "unused.csv", "center_column": "sample_origin"},
        "task": "pdac_vs_rest",
        "features": {
            "biomarkers": ["LYVE1", "REG1B", "TFF1"],
            "demographics": ["age", "sex"],
            "normalize_by_creatinine": True,
            "log_transform": True,
        },
        "model": {
            "name": "logreg",
            "params": {"logreg": {"C": 1.0, "max_iter": 1000},
                       "xgboost": {"n_estimators": 50, "max_depth": 2}},
            "calibrate": False,
        },
        "evaluation": {"split": "leave_one_center_out", "n_bootstrap": 200,
                       "target_specificities": [0.9, 0.95], "random_state": 0},
    }
