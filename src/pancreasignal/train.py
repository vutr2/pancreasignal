"""Experiment runner. (Done — wires your functions together.)

    python -m pancreasignal.train --config configs/urine.yaml
    python -m pancreasignal.train --config configs/urine.yaml --task pdac_vs_benign

    # pdac_vs_control cannot use leave-one-center-out (all controls are from one center):
    python -m pancreasignal.train --task pdac_vs_control --split stratified_kfold

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

from pancreasignal.config import REPORTS_DIR, load_config
from pancreasignal.data.urine import load_urine, select_task
from pancreasignal.evaluation.metrics import compute_metrics
from pancreasignal.evaluation.splits import get_splits
from pancreasignal.features import build_features
from pancreasignal.models import build_model


def run(cfg: dict) -> dict:
    df = load_urine(cfg["data"]["path"])
    rows, y = select_task(df, cfg["task"])
    X = build_features(rows, cfg)
    splits = get_splits(rows, y, cfg)
    if not splits:
        raise RuntimeError(
            f"No usable folds for task {cfg['task']!r} with split {cfg['evaluation']['split']!r} "
            "(see the [LOCO] skip messages above). If no center has both classes, LOCO cannot "
            "evaluate this task: rerun with --split stratified_kfold and report the result as "
            "within-center, not cross-hospital."
        )

    oof = pd.DataFrame({
        "sample_id": rows["sample_id"],
        "center": rows[cfg["data"]["center_column"]],
        "stage": rows["stage"],
        "y": y,
        "ca199": rows["plasma_CA19_9"],
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

    scored = oof.dropna(subset=["prob"])
    results = {
        "task": cfg["task"],
        "model": cfg["model"]["name"],
        "split": cfg["evaluation"]["split"],
        "per_fold": per_fold,
        "model_all": compute_metrics(scored["y"], scored["prob"], cfg),
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.task:
        cfg["task"] = args.task
    if args.model:
        cfg["model"]["name"] = args.model
    if args.split:
        cfg["evaluation"]["split"] = args.split

    out = run(cfg)
    res = out["results"]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg['task']}_{cfg['model']['name']}"
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
    for name, f in res["per_fold"].items():
        print(f"  fold {name:>6}: AUROC {f['auroc']:.3f} (n={f['n']})")
    print(f"  saved -> reports/{run_id}_metrics.json")


if __name__ == "__main__":
    main()
