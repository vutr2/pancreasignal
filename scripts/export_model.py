"""Train the served model on ALL rows of a task and save it for the API (Phase 6).

    python scripts/export_model.py                       # pdac_vs_rest -> models/urine_v1.joblib
    python scripts/export_model.py --task pdac_vs_benign --out models/urine_benign_v1.joblib

Writes two files:
    models/urine_v1.joblib     the model bundle (a pickle; .gitignore keeps it out of git)
    models/urine_v1.meta.json  its SHA-256 + metadata (small and readable: commit this one)

Default task pdac_vs_rest: it learns from all 590 samples (healthy, benign and PDAC), so the
training ranges used for the out-of-distribution check cover the widest set of patients.
Performance numbers do NOT come from this fit: they come from out-of-fold runs (RESULTS.md).
"""
from __future__ import annotations

import argparse

from pancreasignal.api.bundle import build_bundle, save_bundle
from pancreasignal.config import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/urine.yaml")
    ap.add_argument("--task", default="pdac_vs_rest")
    ap.add_argument("--out", default="models/urine_v1.joblib")
    args = ap.parse_args()

    # Step 1: config + task
    cfg = load_config(args.config)
    cfg["task"] = args.task

    # Step 2: fit on all rows and save with its hash
    bundle = build_bundle(cfg)
    sha = save_bundle(bundle, resolve(args.out))

    meta = bundle["meta"]
    print(f"saved {args.out}  (task {meta['task']}, n={meta['n_train']}, PDAC={meta['n_pdac']})")
    print(f"  sha256 {sha}")
    if meta["git"] and meta["git"]["uncommitted_changes"]:
        print("  note: built from uncommitted code; commit and re-export for a reproducible model")


if __name__ == "__main__":
    main()
