"""Phase 1 — these should pass from day one."""
import pytest

from pancreasignal.data.urine import select_task, validate


def test_validate_ok(fake_urine):
    validate(fake_urine)


def test_validate_missing_column(fake_urine):
    with pytest.raises(ValueError, match="Missing columns"):
        validate(fake_urine.drop(columns=["LYVE1"]))


def test_validate_duplicate_ids(fake_urine):
    df = fake_urine.copy()
    df.loc[1, "sample_id"] = df.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate(df)


@pytest.mark.parametrize("task,expected_dx", [
    ("pdac_vs_control", {1, 3}), ("pdac_vs_benign", {2, 3}), ("pdac_vs_rest", {1, 2, 3}),
])
def test_select_task(fake_urine, task, expected_dx):
    rows, y = select_task(fake_urine, task)
    assert set(rows["diagnosis"]) == expected_dx
    assert (y == (rows["diagnosis"] == 3).astype(int)).all()
