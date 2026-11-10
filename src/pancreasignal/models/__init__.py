"""Model builders. ⚠️ YOUR CODE (Phase 2 + 4). Tests: tests/test_models.py

Every model is an sklearn Pipeline so that anything "fitted" (scaler, imputer, calibrator)
only ever sees the training fold. This is the main defense against data leakage.
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data


def build_logreg(params: dict) -> Pipeline:
    """Pipeline: StandardScaler -> LogisticRegression(**params).
    The simplest strong baseline. Its coefficients are easy to explain in an interview.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(**params)),
        ]
    )
    


def build_xgboost(params: dict) -> Pipeline:
    """Pipeline: XGBClassifier(**params, eval_metric='logloss'). No scaler needed for trees."""

    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("xgb", XGBClassifier(**params, eval_metric="logloss")),
        ]
    )


class MinDetectionRate(SelectorMixin, BaseEstimator):
    """Keep features that are "detected" (value > floor) in at least `min_rate` of the
    TRAINING samples. Phase 5 (miRNA): ~62% of values sit on GEO's "not detected" floor.

    Why a Pipeline step and not a pandas filter before training: the detection rate is computed
    from the data, so it must be computed on the training fold only, like the scaler. (It does not
    use the labels, so doing it on all data would leak much less than label-based selection, but
    keeping every data-driven step inside the Pipeline is the simple rule that is never wrong.)
    min_rate = 0 keeps everything (used by the demographics-only baseline).
    """

    def __init__(self, floor: float = 2.612, min_rate: float = 0.5):
        self.floor = floor
        self.min_rate = min_rate

    def fit(self, X, y=None):
        # Step 1: check the input and remember the number (and names) of features
        X = validate_data(self, X)

        # Step 2: share of training samples in which each feature is above the floor
        self.detection_rate_ = (X > self.floor).mean(axis=0)

        # Step 3: keep the features detected often enough (fail if nothing survives)
        self.support_ = self.detection_rate_ >= self.min_rate
        if not self.support_.any():
            raise ValueError(f"No feature is detected in >= {self.min_rate:.0%} of samples")
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_


def build_elasticnet(params: dict) -> Pipeline:
    """Pipeline for p >> n data (Phase 5, miRNA: ~2500 features, a few hundred samples):

        impute -> keep detected features -> scale -> elastic-net logistic regression

    Elastic-net = L1 + L2 penalty. L1 sets most coefficients to exactly 0 (= feature selection);
    L2 keeps groups of correlated miRNAs together instead of picking one at random (pure L1 does).
    LogisticRegressionCV picks the penalty strength C and the L1/L2 mix with its OWN inner CV on
    the training fold, so the outer test fold never influences any choice: this is nested CV.
    """
    # Step 1: missing values (~0.02% here) -> median of the TRAINING fold
    imputer = SimpleImputer(strategy="median")

    # Step 2: drop features that are mostly undetected in the training fold
    detect = MinDetectionRate(floor=params["detection_floor"], min_rate=params["min_detection_rate"])

    # Step 3: elastic-net with inner CV. neg_log_loss is smoother than AUROC on small inner
    #   folds and also rewards well-calibrated probabilities. (sklearn 1.9: l1_ratios alone
    #   selects elastic-net; use_legacy_attributes=False gives scalar C_ / l1_ratio_.)
    clf = LogisticRegressionCV(
        Cs=params["Cs"],
        l1_ratios=params["l1_ratios"],
        cv=params["inner_folds"],
        solver="saga",
        scoring="neg_log_loss",
        max_iter=params["max_iter"],
        n_jobs=-1,
        random_state=0,
        use_legacy_attributes=False,
    )
    return Pipeline([("impute", imputer), ("detect", detect), ("scaler", StandardScaler()), ("clf", clf)])


def build_model(cfg: dict) -> Pipeline:
    """(Done.) Pick the builder from cfg['model']['name'] and wrap with calibration if asked."""
    m = cfg["model"]
    builders = {"logreg": build_logreg, "xgboost": build_xgboost, "elasticnet": build_elasticnet}
    if m["name"] not in builders:
        raise ValueError(f"Unknown model {m['name']!r}")
    model = builders[m["name"]](m["params"][m["name"]])
    if m.get("calibrate"):
        # TODO(Phase 4): wrap `model` in sklearn.calibration.CalibratedClassifierCV
        #   (method="isotonic", cv=5) and return it.
        model = CalibratedClassifierCV(model, method="isotonic", cv=5)
        
    return model
