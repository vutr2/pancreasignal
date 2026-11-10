"""Phase 6 — unit conversion. A wrong factor here silently breaks every prediction."""
import math

import pytest

from pancreasignal.api.units import DATASET_UNITS, allowed_units, normalize_unit, to_dataset_unit


@pytest.mark.parametrize("value,unit,expected_ng_per_ml", [
    (5.0, "ng/mL", 5.0),
    (5.0, "µg/L", 5.0),        # 1 µg/L = 1 ng/mL
    (5000.0, "pg/mL", 5.0),
    (5000.0, "ng/L", 5.0),     # the x1000 trap: ng/L is NOT ng/mL
    (0.005, "µg/mL", 5.0),
    (0.005, "mg/L", 5.0),
])
def test_protein_units(value, unit, expected_ng_per_ml):
    assert to_dataset_unit("LYVE1", value, unit) == pytest.approx(expected_ng_per_ml)


@pytest.mark.parametrize("value,unit,expected_mg_per_ml", [
    (1.0, "mg/mL", 1.0),
    (1.0, "g/L", 1.0),
    (100.0, "mg/dL", 1.0),                 # 100 mg/dL = 1 mg/mL
    (1000.0, "mg/L", 1.0),
    (8.84, "mmol/L", 8.84 * 0.11312),      # creatinine 113.12 g/mol
    (8840.0, "µmol/L", 8.84 * 0.11312),
])
def test_creatinine_units(value, unit, expected_mg_per_ml):
    assert to_dataset_unit("creatinine", value, unit) == pytest.approx(expected_mg_per_ml)


def test_mmol_and_mg_dl_agree():
    # 1 mg/mL of creatinine expressed both ways must give the same number back
    mmol_per_l = 1.0 / 0.11312
    assert to_dataset_unit("creatinine", mmol_per_l, "mmol/L") == pytest.approx(
        to_dataset_unit("creatinine", 100.0, "mg/dL"))


@pytest.mark.parametrize("spelling", ["µg/L", "μg/L", "mcg/L", "ug/l", " UG / L "])
def test_spellings_of_micro(spelling):
    assert normalize_unit(spelling) == "ug/l"   # micro sign, Greek mu, "mc", case, spaces


def test_every_dataset_unit_is_factor_one():
    for analyte, unit in DATASET_UNITS.items():
        assert to_dataset_unit(analyte, 2.5, unit) == 2.5


def test_unknown_unit_is_rejected_with_the_allowed_list():
    with pytest.raises(ValueError, match="Allowed"):
        to_dataset_unit("LYVE1", 1.0, "ng/dL")
    with pytest.raises(ValueError, match="Allowed"):
        to_dataset_unit("creatinine", 1.0, "ng/mL")   # a protein unit is wrong for creatinine
    assert "mmol/L" in allowed_units("creatinine")


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_bad_values(value):
    with pytest.raises(ValueError):
        to_dataset_unit("TFF1", value, "ng/mL")


def test_creatinine_zero_is_rejected():
    with pytest.raises(ValueError, match="creatinine must be > 0"):
        to_dataset_unit("creatinine", 0.0, "mg/mL")
