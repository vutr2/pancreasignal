"""Load + validate the urinary biomarker dataset (Debernardi et al., PLOS Med 2020). (Done.)

Columns (from the dataset documentation):
    sample_id, patient_cohort, sample_origin, age, sex, diagnosis, stage,
    benign_sample_diagnosis, plasma_CA19_9, creatinine, LYVE1, REG1B, TFF1, REG1A
    diagnosis: 1 = control, 2 = benign hepatobiliary disease, 3 = PDAC
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pancreasignal.config import RAW_DIR, resolve

REQUIRED_COLUMNS = [
    "sample_id", "sample_origin", "age", "sex", "diagnosis", "stage",
    "plasma_CA19_9", "creatinine", "LYVE1", "REG1B", "TFF1",
]
DIAGNOSIS = {1: "control", 2: "benign", 3: "pdac"}

# task name -> (negative diagnosis codes, positive diagnosis codes)
TASKS: dict[str, tuple[list[int], list[int]]] = {
    "pdac_vs_control": ([1], [3]),
    "pdac_vs_benign": ([2], [3]),
    "pdac_vs_rest": ([1, 2], [3]),
}


def _find_csv(path: str | Path | None) -> Path:
    if path is not None and resolve(path).exists():
        return resolve(path)
    # fall back: first data CSV in data/raw/urine that isn't the documentation file
    candidates = [p for p in (RAW_DIR / "urine").glob("*.csv") if "documentation" not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(
            "Urine dataset not found. Run: python scripts/download_data.py --only urine"
        )
    return candidates[0]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if the data doesn't look like we expect."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df["sample_id"].is_unique:
        raise ValueError("Duplicate sample_id values")
    bad_dx = set(df["diagnosis"].unique()) - set(DIAGNOSIS)
    if bad_dx:
        raise ValueError(f"Unexpected diagnosis codes: {bad_dx}")
    for col in ["creatinine", "LYVE1", "REG1B", "TFF1"]:
        if df[col].isna().any():
            raise ValueError(f"{col} has missing values (expected complete)")
        if (df[col] < 0).any():
            raise ValueError(f"{col} has negative values")
    return df


def load_urine(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(_find_csv(path))
    return validate(df)


def select_task(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    """Keep only rows relevant to `task`; return (rows, binary label y with 1 = PDAC)."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}. Choose from {list(TASKS)}")
    neg, pos = TASKS[task]
    sub = df[df["diagnosis"].isin(neg + pos)].reset_index(drop=True)
    y = sub["diagnosis"].isin(pos).astype(int).rename("y")
    return sub, y
