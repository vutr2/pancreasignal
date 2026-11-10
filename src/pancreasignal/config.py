"""Project paths and config loading. (Done — no TODOs here.)"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: `base` with `override` on top. Nested dicts are merged key by key,
    everything else (numbers, strings, lists) is replaced."""
    # Step 1: start from a shallow copy so `base` is never modified
    merged = dict(base)

    # Step 2: for each overriding key, merge if both sides are dicts, otherwise replace
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. A config may start with `base: other.yaml` to inherit everything from
    that file and only list what changes (e.g. configs/urine_age_only.yaml).
    Why: copying the whole file means two places to edit; one day they drift apart and two
    "comparable" experiments silently use different evaluation settings."""
    # Step 1: resolve the path relative to the project root
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open() as f:
        cfg = yaml.safe_load(f)

    # Step 2: no `base` key -> plain config, done
    if "base" not in cfg:
        return cfg

    # Step 3: load the parent (relative to this file's folder), then put this file on top
    parent = load_config(path.parent / cfg.pop("base"))
    return _deep_merge(parent, cfg)


def resolve(path: str | Path) -> Path:
    """Turn a config path (relative to project root) into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
