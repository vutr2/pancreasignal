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


# ---------------- Phase 5: fake miRNA table ----------------
def make_fake_mirna(seed: int = 0) -> pd.DataFrame:
    """SMALL RANDOM TABLE shaped like data/geo.load_mirna(): metadata + MIMAT columns.
    miRNA 0-4 are higher in pdac, 5-24 are noise, 25-39 are mostly undetected (floor 2.612)."""
    rng = np.random.default_rng(seed)
    groups = ["control"] * 60 + ["pdac"] * 40 + ["biliary"] * 20 + ["benign"] * 10
    n = len(groups)
    is_pdac = np.array([g == "pdac" for g in groups])
    meta = pd.DataFrame({
        "sample_id": [f"GSM{i}" for i in range(n)],
        "title": [f"{g} X{i}" for i, g in enumerate(groups)],
        "title_prefix": ["S" if i % 2 else "E" for i in range(n)],
        "diagnosis": groups,
        "age": rng.integers(40, 80, n).astype(float),
        "sex": rng.choice(["M", "F"], n),
        "stage": [("IIB" if i % 3 == 0 else "IV") if p else None for i, p in enumerate(is_pdac)],
        "operability": "uncertain",
    })
    meta.loc[0, "age"] = np.nan  # like GEO's "age: uncertain"
    values = rng.normal(6, 1, (n, 40))
    values[:, :5] += 1.5 * is_pdac[:, None]
    values[:, 25:] = np.where(rng.uniform(size=(n, 15)) < 0.9, 2.612, values[:, 25:])
    values = np.maximum(values, 2.612)
    expr = pd.DataFrame(values, columns=[f"MIMAT{i:07d}" for i in range(40)])
    return pd.concat([meta, expr], axis=1)


@pytest.fixture
def fake_mirna() -> pd.DataFrame:
    return make_fake_mirna()


@pytest.fixture
def mirna_cfg() -> dict:
    """configs/mirna.yaml, with a small grid so tests stay fast."""
    from pancreasignal.config import load_config

    cfg = load_config("configs/mirna.yaml")
    cfg["model"]["params"]["elasticnet"].update(Cs=[0.01, 0.1, 1.0], l1_ratios=[0.5], inner_folds=3)
    cfg["evaluation"]["n_bootstrap"] = 50
    return cfg
