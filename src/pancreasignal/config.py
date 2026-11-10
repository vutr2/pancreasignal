"""Project paths and config loading. (Done — no TODOs here.)"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open() as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a config path (relative to project root) into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
