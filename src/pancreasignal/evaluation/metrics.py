"""Clinical evaluation metrics. ⚠️ YOUR CODE (Phase 3). Tests: tests/test_metrics.py

y: 1 = PDAC, 0 = not PDAC.  score: higher = more likely PDAC (probability or raw CA19-9).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


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
    # Step 1: convert inputs to numpy arrays
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)

    # Step 2: validate inputs
    if y.ndim != 1 or prob.ndim != 1 or len(y) != len(prob):
        raise ValueError("y and prob must be 1D arrays of the same length")
    if len(y) == 0:
        raise ValueError("Input arrays must not be empty")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1")
    if not np.all(np.isfinite(prob)) or np.any((prob < 0) | (prob > 1)):
        raise ValueError("prob must be probabilities between 0 and 1")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    # Step 3: put each prediction into an equal-width bin: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    #   prob * n_bins -> 0..n_bins, astype(int) rounds down -> bin number.
    #   prob = 1.0 would get bin n_bins (one past the end), so clip it into the last bin.
    bin_ids = np.minimum((prob * n_bins).astype(int), n_bins - 1)

    # Step 4: for each non-empty bin compare "what the model said" (mean prob) with
    #   "what really happened" (fraction of PDAC), weighted by how many patients are in the bin
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_ids == b
        if not in_bin.any():
            continue
        gap = abs(prob[in_bin].mean() - y[in_bin].mean())
        ece += gap * in_bin.sum() / len(y)

    return float(ece)


def paired_bootstrap_auroc_diff(
    y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_bootstrap: int = 1000,
    alpha: float = 0.05, random_state: int = 42,
) -> dict:
    """AUROC(a) - AUROC(b) on the SAME patients, with a bootstrap confidence interval.

    Why "paired": two models scored on the same patients share the same easy and hard cases, so
    their separate CIs overlap a lot even when one model is clearly better. Resampling the
    patients ONCE and scoring BOTH models on that same sample cancels the shared noise, so the CI
    of the difference is much narrower. If the CI does not contain 0, the difference is real.
    """
    # Step 1: convert inputs to numpy arrays
    y = np.asarray(y)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)

    # Step 2: validate inputs
    if not len(y) == len(score_a) == len(score_b):
        raise ValueError("y, score_a and score_b must have the same length")
    if len(np.unique(y)) < 2:
        raise ValueError("Both classes are required")

    # Step 3: the observed difference on all patients
    diff = roc_auc_score(y, score_a) - roc_auc_score(y, score_b)

    # Step 4: bootstrap: same resampled patients for both models, skip one-class samples
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))

    # Step 5: percentile CI, like bootstrap_auroc_ci
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(diff), "ci95": [float(lo), float(hi)]}


def stage_group(stage) -> str | None:
    """Map a PDAC stage string to "I-II" (early) or "III-IV" (late). None if missing/unknown.

    The dataset mixes detailed and coarse stages ("IA", "IIB", "II", "III", "IV"), so we match
    on the prefix. Check "III"/"IV" FIRST: "III".startswith("I") is also True.
    """
    # Step 1: anything that is not a string (NaN for non-PDAC rows) has no stage
    if not isinstance(stage, str):
        return None

    # Step 2: normalise spelling, then match late stages before early ones
    s = stage.strip().upper()
    if s.startswith(("III", "IV")):
        return "III-IV"
    if s.startswith("I"):
        return "I-II"
    return None


def sensitivity_by_stage(
    y: np.ndarray, score: np.ndarray, stage, specificity: float
) -> dict:
    """Sensitivity at a fixed specificity, separately for early (I-II) and late (III-IV) PDAC.

    The threshold comes from ALL negatives (same as sensitivity_at_specificity), so both groups
    are judged at the same operating point; only the positives are split by stage.
    Why: screening is only useful if it catches EARLY cancers. A model that mostly finds stage
    III-IV tumours (often already symptomatic) helps patients much less.
    """
    # Step 1: convert inputs to numpy arrays and give every row its stage group
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    groups = np.array([stage_group(s) for s in stage], dtype=object)

    # Step 2: validate inputs
    if not len(y) == len(score) == len(groups):
        raise ValueError("y, score and stage must have the same length")

    out = {}
    for group in ["I-II", "III-IV"]:
        # Step 3: keep every negative + only the PDAC cases of this stage group.
        #   The negatives decide the threshold, so it is the same for both groups.
        is_group_pdac = (y == 1) & (groups == group)
        keep = (y == 0) | is_group_pdac
        n_pdac = int(is_group_pdac.sum())

        # Step 4: a group with no PDAC has no sensitivity (report None, not 0)
        if n_pdac == 0:
            out[group] = {"n_pdac": 0, "sensitivity": None}
            continue

        # Step 5: reuse sensitivity_at_specificity on the reduced set
        out[group] = {
            "n_pdac": n_pdac,
            "sensitivity": sensitivity_at_specificity(y[keep], score[keep], specificity),
        }
    return out


def compute_metrics(
    y: np.ndarray, score: np.ndarray, cfg: dict, is_probability: bool = True
) -> dict:
    """(Done.) Bundle every metric for one model's out-of-fold predictions."""
    ev = cfg["evaluation"]
    y, score = np.asarray(y), np.asarray(score)
    lo, hi = bootstrap_auroc_ci(y, score, ev["n_bootstrap"], random_state=ev["random_state"])
    out = {
        "n": len(y),
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
