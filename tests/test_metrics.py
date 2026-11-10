"""Phase 3-4 — fail until you implement evaluation/metrics.py."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pancreasignal.evaluation.metrics import (
    bootstrap_auroc_ci,
    expected_calibration_error,
    sensitivity_at_specificity,
)


def test_sens_at_spec_perfect():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    assert sensitivity_at_specificity(y, s, 0.95) == pytest.approx(1.0)


def test_sens_at_spec_known():
    # negatives 0..99, positives 50..149 -> at 90% spec threshold ~90, positives >= 90 = 60/100
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([np.arange(100), np.arange(50, 150)]).astype(float)
    assert sensitivity_at_specificity(y, s, 0.90) == pytest.approx(0.60, abs=0.02)


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    lo, hi = bootstrap_auroc_ci(y, s, n_bootstrap=300, random_state=0)
    auc = roc_auc_score(y, s)
    assert lo < auc < hi
    assert 0 < hi - lo < 0.2
    assert bootstrap_auroc_ci(y, s, 300, random_state=0) == (lo, hi), "must be reproducible"


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)        # perfectly calibrated
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.full_like(p, 0.9)) > 0.3   # overconfident


# ---------------- Phase 4: sensitivity by stage ----------------
from pancreasignal.evaluation.metrics import sensitivity_by_stage, stage_group


def test_stage_group():
    assert [stage_group(s) for s in ["IA", "IIB", "II", "III", "IIIB", "IV"]] == \
        ["I-II", "I-II", "I-II", "III-IV", "III-IV", "III-IV"]
    assert stage_group(np.nan) is None, "non-PDAC rows have no stage"


def test_sensitivity_by_stage():
    # negatives score 0..99 -> at 90% specificity the threshold sits just above ~89.
    # early PDAC all score 200 (caught), late PDAC all score 50 (missed)
    y = np.array([0] * 100 + [1] * 10 + [1] * 10)
    s = np.concatenate([np.arange(100), np.full(10, 200), np.full(10, 50)]).astype(float)
    stage = [np.nan] * 100 + ["IIB"] * 10 + ["IV"] * 10
    out = sensitivity_by_stage(y, s, stage, 0.90)
    assert out["I-II"] == {"n_pdac": 10, "sensitivity": 1.0}
    assert out["III-IV"] == {"n_pdac": 10, "sensitivity": 0.0}


def test_sensitivity_by_stage_empty_group():
    y = np.array([0] * 50 + [1] * 10)
    s = np.concatenate([np.arange(50), np.full(10, 100)]).astype(float)
    out = sensitivity_by_stage(y, s, [np.nan] * 50 + ["II"] * 10, 0.95)
    assert out["III-IV"] == {"n_pdac": 0, "sensitivity": None}


# ---------------- Phase 4: paired comparison ----------------
from pancreasignal.evaluation.metrics import paired_bootstrap_auroc_diff


def test_paired_diff_same_model_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 1, 200)
    out = paired_bootstrap_auroc_diff(y, s, s, n_bootstrap=200)
    assert out == {"diff": 0.0, "ci95": [0.0, 0.0]}


def test_paired_diff_detects_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.5, 400)
    bad = y + rng.normal(0, 3, 400)
    out = paired_bootstrap_auroc_diff(y, good, bad, n_bootstrap=300)
    assert out["diff"] > 0 and out["ci95"][0] > 0, "CI must exclude 0 for a clearly better model"
