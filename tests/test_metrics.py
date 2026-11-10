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
