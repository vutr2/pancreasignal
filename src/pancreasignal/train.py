"""Experiment runner. (Done — wires your functions together.)

    python -m pancreasignal.train --config configs/urine.yaml
    python -m pancreasignal.train --config configs/urine.yaml --task pdac_vs_benign

    # pdac_vs_control cannot use leave-one-center-out (all controls are from one center):
    python -m pancreasignal.train --task pdac_vs_control --split stratified_kfold

    # Phase 4 variants
    python -m pancreasignal.train --task pdac_vs_benign --model xgboost
    python -m pancreasignal.train --task pdac_vs_benign --calibrate          # isotonic calibration
    python -m pancreasignal.train --config configs/urine_age_only.yaml --task pdac_vs_benign

Outputs (in reports/):
    <run_id>_metrics.json   all metrics + config
    <run_id>_oof.csv        out-of-fold predictions per sample (for plots, stage analysis)
Also logs to MLflow if it is installed (`pip install -e ".[tracking]"`).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from pancreasignal.config import REPORTS_DIR, load_config
from pancreasignal.data.geo import load_mirna, select_mirna_task
from pancreasignal.data.urine import load_urine, select_task
from pancreasignal.evaluation.metrics import compute_metrics, sensitivity_by_stage
from pancreasignal.evaluation.splits import get_splits
from pancreasignal.features import build_features, build_mirna_features
from pancreasignal.models import build_model


def load_task_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(rows, X, y) for cfg['task'] on cfg['data']['dataset']: 'urine' (default) or 'mirna'."""
    # Step 1: which dataset? `dataset` is optional: configs from Phase 1-4 have none -> urine
    dataset = cfg["data"].get("dataset", "urine")

    # Step 2: load, keep the rows of this task, build the feature matrix
    if dataset == "urine":
        df = load_urine(cfg["data"]["path"])
        rows, y = select_task(df, cfg["task"])
        X = build_features(rows, cfg)
    elif dataset == "mirna":
        df = load_mirna(cfg["data"]["path"])
        rows, y = select_mirna_task(df, cfg["task"])
        X = build_mirna_features(rows, cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose 'urine' or 'mirna'.")
    return rows, X, y


def _n_nonzero_coefs(model) -> int | None:
    """How many features a linear model really uses (non-zero coefficients). None for trees."""
    final = model[-1] if isinstance(model, Pipeline) else None
    if final is None or not hasattr(final, "coef_"):
        return None
    return int(np.count_nonzero(final.coef_))


def run(cfg: dict) -> dict:
    rows, X, y = load_task_data(cfg)
    splits = get_splits(rows, y, cfg)
    if not splits:
        raise RuntimeError(
            f"No usable folds for task {cfg['task']!r} with split {cfg['evaluation']['split']!r} "
            "(see the [LOCO] skip messages above). If no center has both classes, LOCO cannot "
            "evaluate this task: rerun with --split stratified_kfold and report the result as "
            "within-center, not cross-hospital."
        )

    # Columns that only some datasets have (miRNA: no hospital, no CA19-9) -> NaN / "all"
    center_column = cfg["data"].get("center_column")
    oof = pd.DataFrame({
        "sample_id": rows["sample_id"],
        "center": rows[center_column] if center_column else "all",
        "stage": rows.get("stage", np.nan),
        "y": y,
        "ca199": rows.get("plasma_CA19_9", np.nan),
        "fold": None,
        "prob": np.nan,
    })
    per_fold = {}
    for name, tr, te in splits:
        model = build_model(cfg)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        oof.loc[oof.index[te], "prob"] = p
        oof.loc[oof.index[te], "fold"] = name
        if y.iloc[te].nunique() == 2:
            per_fold[name] = {"n": len(te), "auroc": float(roc_auc_score(y.iloc[te], p))}
            # p >> n: how many features did this fold's model keep? (varies a lot between folds)
            n_used = _n_nonzero_coefs(model)
            if n_used is not None:
                per_fold[name]["n_nonzero_coefs"] = n_used

    scored = oof.dropna(subset=["prob"])
    results = {
        "dataset": cfg["data"].get("dataset", "urine"),
        "task": cfg["task"],
        "model": cfg["model"]["name"],
        # `name` is optional (old configs/tests have none), so .get with a default is right here
        "name": cfg.get("name", "urine_panel"),
        "calibrated": bool(cfg["model"].get("calibrate")),
        "n_features": int(X.shape[1]),
        # the full list only when it is short (the miRNA model has ~2500 columns)
        "features": list(X.columns) if X.shape[1] <= 50 else None,
        "split": cfg["evaluation"]["split"],
        "per_fold": per_fold,
        "model_all": compute_metrics(scored["y"], scored["prob"], cfg),
    }
    # Sensitivity for early (I-II) vs late (III-IV) PDAC, at each target specificity.
    # Early-stage sensitivity is what matters for screening.
    results["sens_by_stage"] = {
        f"spec{int(spec * 100)}": sensitivity_by_stage(
            scored["y"], scored["prob"], scored["stage"], spec
        )
        for spec in cfg["evaluation"]["target_specificities"]
    }
    # Fair head-to-head vs CA19-9: same rows (only those with a CA19-9 measurement)
    sub = scored.dropna(subset=["ca199"])
    if sub["y"].nunique() == 2 and len(sub) >= 20:
        results["model_on_ca199_subset"] = compute_metrics(sub["y"], sub["prob"], cfg)
        results["ca199_baseline"] = compute_metrics(
            sub["y"], sub["ca199"], cfg, is_probability=False
        )
    return {"results": results, "oof": oof}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/urine.yaml")
    ap.add_argument("--task", help="override cfg['task']")
    ap.add_argument("--model", help="override cfg['model']['name']")
    ap.add_argument("--split", choices=["leave_one_center_out", "stratified_kfold"],
                    help="override cfg['evaluation']['split']")
    ap.add_argument("--calibrate", action="store_true",
                    help="wrap the model in isotonic calibration (sets cfg['model']['calibrate'])")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.task:
        cfg["task"] = args.task
    if args.model:
        cfg["model"]["name"] = args.model
    if args.split:
        cfg["evaluation"]["split"] = args.split
    if args.calibrate:
        cfg["model"]["calibrate"] = True

    out = run(cfg)
    res = out["results"]
    # File name = time + task + model + feature set (+ "_calibrated"), e.g.
    #   20261111-101500_pdac_vs_benign_logreg_age_only_metrics.json
    # so a baseline or a calibrated run can never be mistaken for the main model.
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['task']}_{cfg['model']['name']}_{res['name']}"
    if res["calibrated"]:
        run_id += "_calibrated"
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f"{run_id}_metrics.json").write_text(json.dumps({"config": cfg, **res}, indent=2))
    out["oof"].to_csv(REPORTS_DIR / f"{run_id}_oof.csv", index=False)

    try:
        import mlflow

        mlflow.set_experiment("pancreasignal-urine")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params({"task": cfg["task"], "model": cfg["model"]["name"],
                               "split": cfg["evaluation"]["split"]})
            for k, v in res["model_all"].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_artifact(str(REPORTS_DIR / f"{run_id}_metrics.json"))
    except ImportError:
        pass

    m = res["model_all"]
    print(f"\n{run_id}  (split: {cfg['evaluation']['split']})")
    print(f"  model  AUROC {m['auroc']:.3f}  (95% CI {m['auroc_ci95'][0]:.3f}-{m['auroc_ci95'][1]:.3f})"
          f"  n={m['n']}  PDAC={m['n_pdac']}")
    if "ca199_baseline" in res:
        a, b = res["model_on_ca199_subset"], res["ca199_baseline"]
        print(f"  on CA19-9 subset (n={a['n']}): model {a['auroc']:.3f} vs CA19-9 {b['auroc']:.3f}")
    if m.get("ece") is not None:
        print(f"  calibration: Brier {m['brier']:.3f}  ECE {m['ece']:.3f}")
    for spec_key, groups in res["sens_by_stage"].items():
        parts = [f"{g} {v['sensitivity']:.2f} (n={v['n_pdac']})" for g, v in groups.items()
                 if v["sensitivity"] is not None]
        print(f"  sensitivity by stage @ {spec_key[4:]}% spec: " + ", ".join(parts))
    for name, f in res["per_fold"].items():
        print(f"  fold {name:>6}: AUROC {f['auroc']:.3f} (n={f['n']})")
    print(f"  saved -> reports/{run_id}_metrics.json")


if __name__ == "__main__":
    main()
