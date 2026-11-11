"""Build, save, load and use the model bundle served by the API (Phase 6). Tests: tests/test_api.py

A bundle = the fitted Pipeline + what the API needs to use it safely:
  - the config, so serving builds features with the SAME build_features as training
  - training ranges (out-of-distribution check) and quantiles (to explain a value as a percentile)
  - metadata: task, data hash, git commit, library versions, out-of-fold performance from reports/

Security: a .joblib file is a pickle, and loading a pickle can run arbitrary code. Only load bundles
you built yourself. load_bundle() refuses a file whose SHA-256 differs from the one save_bundle()
wrote, so a swapped or corrupted file is never unpickled.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from pancreasignal.api.light import UNIT_ERROR_FACTOR, check_ranges, percentile  # noqa: F401 (re-export)
from pancreasignal.config import PROJECT_ROOT
from pancreasignal.features import build_features
from pancreasignal.models import build_model
from pancreasignal.train import load_task_data

BUNDLE_VERSION = 1
RAW_INPUTS = ["LYVE1", "REG1B", "TFF1", "creatinine", "age"]   # checked against training ranges
# UNIT_ERROR_FACTOR, percentile() and check_ranges() live in api/light.py: shared with the pure-Python
# copy of this model (browser demo), so both apply exactly the same checks.


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> dict | None:
    """Commit the bundle was built from, and whether there were uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                               text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": sha, "uncommitted_changes": dirty}


def _performance_reference(task: str) -> dict | None:
    """Out-of-fold performance of this model+task from reports/ (what users should expect)."""
    from pancreasignal.reporting import get_run, load_runs  # optional: reports/ may not exist

    try:
        runs = load_runs()
        run = get_run(runs[runs["dataset"] == "urine"], task, "logreg")
    except (FileNotFoundError, KeyError):
        return None
    m = run["results"]["model_all"]
    return {"run_id": run["run_id"], "split": run["split"], "auroc": m["auroc"], "auroc_ci95": m["auroc_ci95"],
            "sens_at_spec90": m["sens_at_spec90"], "sens_at_spec95": m["sens_at_spec95"]}


def build_bundle(cfg: dict) -> dict:
    """Fit the served model ONCE on all rows of cfg['task'] and collect what the API needs."""
    # Step 1: only the recommended model (Phase 4): uncalibrated logistic regression, urine panel.
    #   Its reasons are exact (see explain_row) and it was the most robust across hospitals.
    if (cfg["model"]["name"] != "logreg" or cfg["model"].get("calibrate")
            or cfg["data"].get("dataset", "urine") != "urine"):
        raise ValueError("The API serves the urine logistic regression only (model.name = 'logreg', no calibration)")

    # Step 2: same data and features as train.run; fit on every row (evaluation was done out-of-fold)
    rows, X, y = load_task_data(cfg)
    model = build_model(cfg).fit(X, y)

    # Step 3: what "normal" looked like in training, for the out-of-distribution check and percentiles
    ranges = {
        "inputs": {c: [float(rows[c].min()), float(rows[c].max())] for c in RAW_INPUTS},
        "features": {c: [float(X[c].min()), float(X[c].max())] for c in X.columns},
    }
    quantiles = {c: np.quantile(X[c], np.linspace(0, 1, 101)).tolist() for c in X.columns}

    # Step 4: metadata, so anyone can see what exactly is being served
    manifest = PROJECT_ROOT / "data" / "manifest.json"
    meta = {
        "name": cfg.get("name", "urine_panel"),
        "task": cfg["task"],
        "features": list(X.columns),
        "n_train": len(y),
        "n_pdac": int(y.sum()),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_files_sha256": json.loads(manifest.read_text())["urine"]["files"] if manifest.exists() else None,
        "git": _git_commit(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "performance_reference": _performance_reference(cfg["task"]),
    }
    return {"bundle_version": BUNDLE_VERSION, "model": model, "cfg": copy.deepcopy(cfg),
            "ranges": ranges, "quantiles": quantiles, "meta": meta}


def save_bundle(bundle: dict, path: str | Path) -> str:
    """Write <name>.joblib (never commit it) and <name>.meta.json with its SHA-256 (commit this one)."""
    # Step 1: write the pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    # Step 2: hash it and write a readable JSON next to it
    sha = _sha256(path)
    meta = {"file": path.name, "sha256": sha, **bundle["meta"]}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return sha


def load_bundle(path: str | Path) -> dict:
    """Load a bundle, but only if its SHA-256 matches <name>.meta.json."""
    # Step 1: check the hash BEFORE unpickling (unpickling an untrusted file can run code)
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Need {path} and {meta_path}. Run: python scripts/export_model.py")
    expected = json.loads(meta_path.read_text())["sha256"]
    if _sha256(path) != expected:
        raise ValueError(f"SHA-256 of {path.name} does not match {meta_path.name}: refusing to load it")

    # Step 2: unpickle and check the format version
    bundle = joblib.load(path)
    if bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"Bundle version {bundle.get('bundle_version')} != {BUNDLE_VERSION}: re-export the model")

    # Step 3: a pickled sklearn model may behave differently under another sklearn version
    if bundle["meta"]["sklearn_version"] != sklearn.__version__:
        warnings.warn(f"Model built with scikit-learn {bundle['meta']['sklearn_version']}, "
                      f"running {sklearn.__version__}: re-export to be safe", stacklevel=2)
    return bundle


def predict_row(bundle: dict, inputs: dict[str, float], age: float, sex: str) -> dict:
    """Predict one patient. `inputs` must already be in dataset units (proteins ng/mL,
    creatinine mg/mL; see api/units.py). Returns probability, every feature's contribution,
    out-of-range inputs and warnings."""
    model, cfg = bundle["model"], bundle["cfg"]

    # Step 1: one raw row, then the SAME build_features as training (no training/serving skew)
    raw = pd.DataFrame([{**inputs, "age": age, "sex": sex}])
    X = build_features(raw, cfg)
    probability = float(model.predict_proba(X)[0, 1])

    # Step 2: reasons. For a linear model on standardised features, SHAP is exact and simple:
    #   contribution = coefficient x standardised value (the training mean of every feature is 0),
    #   and intercept + sum(contributions) = log-odds of the prediction.
    z = model.named_steps["scaler"].transform(X)[0]
    coef = model.named_steps["logreg"].coef_[0]
    contributions = [
        {"feature": f, "model_value": float(X.iloc[0][f]), "contribution_log_odds": float(c * zz),
         # a percentile means nothing for a 0/1 feature
         "training_percentile": None if f == "sex" else percentile(bundle["quantiles"][f], float(X.iloc[0][f]))}
        for f, c, zz in zip(X.columns, coef, z, strict=True)
    ]
    contributions.sort(key=lambda d: abs(d["contribution_log_odds"]), reverse=True)

    # Step 3: out-of-range inputs/features and likely unit mistakes (shared with api/light.py)
    out_of_range, warnings_ = check_ranges(bundle["ranges"], inputs, age,
                                           {f: float(X.iloc[0][f]) for f in X.columns})

    return {"probability": probability,
            "base_log_odds": float(model.named_steps["logreg"].intercept_[0]),
            "contributions": contributions,
            "out_of_range": out_of_range,
            "warnings": warnings_}
