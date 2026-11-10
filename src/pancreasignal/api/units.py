"""Unit conversion for the API (Phase 6). Tests: tests/test_units.py

The model was trained on the units of the Debernardi 2020 dataset:
    LYVE1, REG1B, TFF1 : ng/mL
    creatinine         : mg/mL   (unusual: labs usually report mg/dL, mmol/L or µmol/L)

Every value must be converted to these units BEFORE the model sees it. A silent x1000 mistake
(ng/L instead of ng/mL) would give a confident but wrong prediction, so unknown units are
rejected instead of guessed.
"""
from __future__ import annotations

import math

CREATININE_MOLAR_MASS = 113.12  # g/mol, to convert mmol/L and µmol/L

# unit as people write it -> multiply by this to get the dataset unit
PROTEIN_UNITS = {           # -> ng/mL
    "ng/mL": 1.0,
    "µg/L": 1.0,            # 1 µg/L = 1 ng/mL
    "pg/mL": 1e-3,
    "ng/L": 1e-3,           # the classic trap: ng/L is 1000x smaller than ng/mL
    "µg/mL": 1e3,
    "mg/L": 1e3,
}
CREATININE_UNITS = {        # -> mg/mL
    "mg/mL": 1.0,
    "g/L": 1.0,
    "mg/dL": 1e-2,          # 100 mg/dL = 1 mg/mL
    "mg/L": 1e-3,
    "mmol/L": CREATININE_MOLAR_MASS / 1e3,   # 1 mmol/L = 113.12 mg/L = 0.11312 mg/mL
    "µmol/L": CREATININE_MOLAR_MASS / 1e6,
}
DATASET_UNITS = {"LYVE1": "ng/mL", "REG1B": "ng/mL", "TFF1": "ng/mL", "creatinine": "mg/mL"}
_TABLES = {"LYVE1": PROTEIN_UNITS, "REG1B": PROTEIN_UNITS, "TFF1": PROTEIN_UNITS,
           "creatinine": CREATININE_UNITS}


def normalize_unit(unit: str) -> str:
    """Make spellings comparable: 'µg/L', 'μg/l', 'mcg/L', ' UG / L ' all become 'ug/l'."""
    # Step 1: trim and lowercase
    u = unit.strip().lower()

    # Step 2: three ways to write "micro": µ (micro sign), μ (Greek mu), mc (as in mcg)
    for micro in ("µ", "μ", "mc"):
        u = u.replace(micro, "u")

    # Step 3: ignore spaces ("ng / mL")
    return u.replace(" ", "")


def allowed_units(analyte: str) -> list[str]:
    """The units accepted for one analyte, as shown to users."""
    if analyte not in _TABLES:
        raise ValueError(f"Unknown analyte {analyte!r}. Known: {sorted(_TABLES)}")
    return list(_TABLES[analyte])


def to_dataset_unit(analyte: str, value: float, unit: str) -> float:
    """Convert one measurement to the unit the model was trained on (see DATASET_UNITS)."""
    # Step 1: which conversion table?
    if analyte not in _TABLES:
        raise ValueError(f"Unknown analyte {analyte!r}. Known: {sorted(_TABLES)}")
    table = {normalize_unit(u): factor for u, factor in _TABLES[analyte].items()}

    # Step 2: the value must be a real, non-negative number; creatinine is a denominator -> > 0
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{analyte} must be a finite number >= 0, got {value}")
    if analyte == "creatinine" and value == 0:
        raise ValueError("creatinine must be > 0 (every protein is divided by it)")

    # Step 3: look up the unit; an unknown unit is an error, never a guess
    key = normalize_unit(unit)
    if key not in table:
        raise ValueError(f"Unknown unit {unit!r} for {analyte}. Allowed: {allowed_units(analyte)}")
    return value * table[key]
