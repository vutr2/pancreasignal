"""Clinical evaluation metrics. ⚠️ YOUR CODE (Phase 3). Tests: tests/test_metrics.py

y: 1 = PDAC, 0 = not PDAC.  score: higher = more likely PDAC (probability or raw CA19-9).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.metrics import roc_curve


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
    #convert inputs to numpy arrays
    y = np.asarray(y)
    score = np.asarray(score)

    #step 2: Validate inputs
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score):
        raise ValueError("y and score must be 1D arrays of the same length")

    if not 0 <= specificity <= 1:
        raise ValueError("specificity must be between 0 and 1")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be binary (0 or 1)")
    if not np.all(np.isfinite(score)):
        raise ValueError("score must be finite numbers")
    if len(np.unique(y)) != 2:
        raise ValueError("Both positive and negative casses are required")

    #step 3: Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)

    # Step 4: Find thresholds satisfying specificity constraint
    valid = fpr <= (1 - specificity)

    # Step 5: Find maximum sensitivity among valid thresholds
    best_idx = np.argmax(tpr[valid])

    # Step 6: Get the corresponding threshold
    best_threshold = thresholds[valid][best_idx]

    # Step 7: Calculate sensitivity
    sensitivity = np.mean(score[y == 1] >= best_threshold)

    return float(sensitivity)



def bootstrap_auroc_ci(
    y: np.ndarray, score: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """95% confidence interval for AUROC by bootstrap.

    Recipe: repeat n_bootstrap times -> sample n rows WITH replacement -> skip the sample if it
    has only one class -> compute AUROC. Return the (alpha/2, 1 - alpha/2) percentiles.
    Use np.random.default_rng(random_state) so results are reproducible.
    """

    #step 1 convert inputs to numpy arrays
    y = np.asanyarray(y)
    score = np.asarray(score)

    #Step2: Validate Inputs
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score):
        raise ValueError("y and score must be 1D arrays of the same length")

    if len(y) == 0:
        raise ValueError("Input arrays must not be empty")
    if not np.all(np.isin(y, [0,1])):
        raise ValueError("y must contain only in 0 and 1")
    if not np.all(np.isfinite(score)):
        raise ValueError('score must contain only finite numbers')
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes are required")
    if not 0< alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    #Step 3: Initialize random generator
    rng = np.random.default_rng(random_state)

    #Step 4: Store boostrap AUROC values
    auroc_scores = []

    #Step 5: Boostrap loop through sampling
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y), size=len(y), replace=True)
        y_sample = y[indices]
        score_sample = score[indices]

        #Skip smaples contains only one class
        if len(np.unique(y_sample)) < 2:
            continue

        #Calculate AUROC
        auc = roc_auc_score(y_sample, score_sample)
        auroc_scores.append(auc)
    #Step 6: Ensure enough valid boostrap samples
    if len(auroc_scores) == 0:
        raise ValueError("No valid bootstrap samples were generated. Increase n_bootstrap or check your data.")
    #Step 7 Calculate confidence interval
    lower = np.percentile(auroc_scores, 100 * alpha / 2)
    upper = np.percentile(auroc_scores, 100 * (1 - alpha / 2))
    return float(lower), float(upper)

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
