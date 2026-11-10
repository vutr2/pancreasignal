"""Clinical evaluation metrics. ⚠️ YOUR CODE (Phase 3). Tests: tests/test_metrics.py

y: 1 = PDAC, 0 = not PDAC.  score: higher = more likely PDAC (probability or raw CA19-9).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def sensitivity_at_specificity(y: np.ndarray, score: np.ndarray, specificity: float) -> float:
    """Sensitivity when the threshold is set so that specificity >= `specificity`.

    Recipe:
      1. thresholds come from the NEGATIVE cases' scores: pick the threshold t so that at
         least `specificity` of negatives have score < t ... (hint: np.quantile on negatives,
         or sklearn.metrics.roc_curve and take the best TPR where FPR <= 1 - specificity)
      2. sensitivity = fraction of positives with score >= t
    Why: in screening you fix how many healthy people you are willing to alarm (e.g. 5%),
    then ask how many cancers you catch. AUROC alone hides this.
    """
    raise NotImplementedError("TODO(Phase 3): implement sensitivity_at_specificity")


def bootstrap_auroc_ci(
    y: np.ndarray, score: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """95% confidence interval for AUROC by bootstrap.

    Recipe: repeat n_bootstrap times -> sample n rows WITH replacement -> skip the sample if it
    has only one class -> compute AUROC. Return the (alpha/2, 1 - alpha/2) percentiles.
    Use np.random.default_rng(random_state) so results are reproducible.
    """
    raise NotImplementedError("TODO(Phase 3): implement bootstrap_auroc_ci")


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: split [0,1] into n_bins equal-width bins; for each non-empty bin,
    |mean(prob) - mean(y)| weighted by (bin size / n). Sum over bins.
    Why: if the model says "30% risk", about 30% of those patients should have PDAC.
    """
    raise NotImplementedError("TODO(Phase 4): implement expected_calibration_error")


def compute_metrics(
    y: np.ndarray, score: np.ndarray, cfg: dict, is_probability: bool = True
) -> dict:
    """(Done.) Bundle every metric for one model's out-of-fold predictions."""
    ev = cfg["evaluation"]
    y, score = np.asarray(y), np.asarray(score)
    lo, hi = bootstrap_auroc_ci(y, score, ev["n_bootstrap"], random_state=ev["random_state"])
    out = {
        "n": int(len(y)),
        "n_pdac": int(y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "auroc_ci95": [float(lo), float(hi)],
    }
    for spec in ev["target_specificities"]:
        out[f"sens_at_spec{int(spec * 100)}"] = float(sensitivity_at_specificity(y, score, spec))
    if is_probability:
        out["brier"] = float(brier_score_loss(y, score))
        try:
            out["ece"] = float(expected_calibration_error(y, score))
        except NotImplementedError:
            out["ece"] = None  # Phase 4
    return out
