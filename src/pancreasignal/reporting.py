"""Read finished runs from reports/ (Phase 4). Tests: tests/test_reporting.py

Used by scripts/make_results.py (RESULTS.md, README, MODEL_CARD) and notebooks/02_results.ipynb,
so every table and plot is built from the saved run files, never from hand-typed numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pancreasignal.config import REPORTS_DIR

# Short key -> label used in tables, in the order rows should appear
VARIANTS = {
    "age_only": "Age only (logistic regression)",
    "logreg": "Logistic regression",
    "logreg_calibrated": "Logistic regression + isotonic calibration",
    "xgboost": "XGBoost",
    "xgboost_calibrated": "XGBoost + isotonic calibration",
}
TASKS = {
    "pdac_vs_control": "PDAC vs control",
    "pdac_vs_benign": "PDAC vs benign",
    "pdac_vs_rest": "PDAC vs rest",
}


def variant_of(results: dict) -> str:
    """Give a run a short key such as 'age_only', 'logreg', 'logreg_calibrated', 'xgboost'."""
    # Step 1: a baseline feature set is called by its own name (e.g. "age_only").
    #   Reports made before Phase 4 have no "name" -> they used the main urine panel.
    name = results.get("name", "urine_panel")
    if name != "urine_panel":
        return name

    # Step 2: the main urine panel is called by its model, plus "_calibrated" if calibrated
    key = results["model"]
    if results.get("calibrated"):
        key += "_calibrated"
    return key


def load_runs(reports_dir: str | Path = REPORTS_DIR) -> pd.DataFrame:
    """One row per (task, variant), keeping only the NEWEST run.

    Columns: run_id, task, variant, split, results (the whole metrics dict), oof_path.
    """
    # Step 1: read every metrics file
    rows = []
    for path in sorted(Path(reports_dir).glob("*_metrics.json")):
        data = json.loads(path.read_text())
        run_id = path.name.removesuffix("_metrics.json")
        rows.append({
            "run_id": run_id,
            "task": data["task"],
            "variant": variant_of(data),
            "split": data["split"],
            "results": data,
            "oof_path": path.with_name(f"{run_id}_oof.csv"),
        })
    if not rows:
        raise FileNotFoundError(f"No *_metrics.json in {reports_dir}. Run pancreasignal.train first.")

    # Step 2: run_id starts with a timestamp (YYYYMMDD-HHMMSS), so sorting by it = oldest first
    runs = pd.DataFrame(rows).sort_values("run_id")

    # Step 3: keep the newest run of each (task, variant); older reruns are ignored
    return runs.drop_duplicates(["task", "variant"], keep="last").reset_index(drop=True)


def paired_oof(runs: pd.DataFrame, task: str, variant_a: str, variant_b: str) -> pd.DataFrame:
    """Out-of-fold predictions of two variants side by side, on the patients BOTH scored.

    Columns: sample_id, center, stage, y, ca199, prob_a, prob_b.
    """
    # Step 1: read both out-of-fold files
    a = pd.read_csv(get_run(runs, task, variant_a)["oof_path"])
    b = pd.read_csv(get_run(runs, task, variant_b)["oof_path"])

    # Step 2: join on sample_id (never on row order) and drop patients either model skipped
    merged = a.merge(b[["sample_id", "prob"]], on="sample_id", suffixes=("_a", "_b"))
    return merged.dropna(subset=["prob_a", "prob_b"]).reset_index(drop=True)


def get_run(runs: pd.DataFrame, task: str, variant: str) -> dict:
    """The newest run for (task, variant) as a dict. KeyError with a hint if it was never run."""
    match = runs[(runs["task"] == task) & (runs["variant"] == variant)]
    if match.empty:
        raise KeyError(f"No run for task={task!r}, variant={variant!r}. Run it with pancreasignal.train.")
    return match.iloc[0].to_dict()
