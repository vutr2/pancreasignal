"""Pure-Python copy of the served urine model, for places without scikit-learn, numpy or pandas: the
free static Hugging Face Space runs it inside the visitor's browser (Pyodide). Tests: tests/test_light_model.py

The served model is: features (log1p(marker / creatinine), age, sex 0/1) -> StandardScaler ->
LogisticRegression. That is a handful of numbers, so export_light() writes them to JSON and
predict_row_light() redoes the arithmetic. It must give the SAME output as api/bundle.predict_row
(tested against it); the percentile, out-of-range and unit-mistake checks below are shared code, used
by both, so they cannot drift apart.

Why not just load the .joblib in the browser? It is a pickle: it needs the exact scikit-learn that
wrote it, and unpickling can run code. JSON numbers need neither.
"""
from __future__ import annotations

import bisect
import math

LIGHT_FORMAT = "pancreasignal-light-v1"
# A value 10x above anything seen in training is more likely a unit mistake than a real patient.
# 10, not 100: a typical LYVE1 (~1.2 ng/mL) sent 1000x too high is 1200, which is above 10x the
# training max (23.9) but BELOW 100x it, so 100 would miss the most common mistake.
UNIT_ERROR_FACTOR = 10


# ---------------------------------------------------------------- shared with api/bundle.predict_row
def percentile(quantiles, value: float) -> float:
    """Where `value` falls in the training distribution, 0-100 (= np.searchsorted(q, v, 'right'), clipped)."""
    return float(min(max(bisect.bisect_right(quantiles, value), 1), 101) - 1)


def check_ranges(ranges: dict, inputs: dict[str, float], age: float, features: dict[str, float]) -> tuple[list, list]:
    """(out_of_range, warnings) for one patient: raw inputs and model features outside the training
    min..max, and values so far outside (10x) that a wrong unit is the likely cause."""
    raw = {**inputs, "age": age}

    # Step 1: out-of-distribution = any raw input or model feature outside the training min..max
    checks = [(name, raw[name], lo, hi) for name, (lo, hi) in ranges["inputs"].items()]
    checks += [(f"{f} (model feature)", features[f], lo, hi)
               for f, (lo, hi) in ranges["features"].items() if f != "sex"]
    out_of_range = [{"feature": n, "value": v, "training_min": lo, "training_max": hi}
                    for n, v, lo, hi in checks if not lo <= v <= hi]

    # Step 2: a value 10x outside training is more often a unit mistake than a real patient.
    #   Limitation: only "too high" mistakes show up for the proteins. Their training minimum is
    #   ~0, so a value 1000x too LOW (ng/L sent as ng/mL) still looks normal. Units must be right.
    warnings_ = []
    for name, (lo, hi) in ranges["inputs"].items():
        value = raw[name]
        if name != "age" and (value > hi * UNIT_ERROR_FACTOR or value < lo / UNIT_ERROR_FACTOR):
            warnings_.append(f"{name} = {value:g} is more than {UNIT_ERROR_FACTOR}x outside the training range "
                             f"({lo:g} to {hi:g}): check the unit.")
    if out_of_range:
        warnings_.append("Some inputs are outside the training data: this prediction is an extrapolation.")
    return out_of_range, warnings_


# ---------------------------------------------------------------- export + predict
def export_light(bundle: dict) -> dict:
    """JSON-safe copy of a full bundle (fitted Pipeline) with only what predict_row_light needs."""
    # Step 1: only the pipeline this file knows how to recompute
    model = bundle["model"]
    if list(model.named_steps) != ["scaler", "logreg"]:
        raise ValueError(f"export_light supports scaler -> logreg only, got {list(model.named_steps)}")
    scaler, logreg = model.named_steps["scaler"], model.named_steps["logreg"]
    order = list(bundle["meta"]["features"])
    if list(scaler.feature_names_in_) != order:
        raise ValueError(f"Feature order mismatch: {list(scaler.feature_names_in_)} vs {order}")

    # Step 2: plain floats and lists only
    def floats(pairs: dict) -> dict:
        return {k: [float(lo), float(hi)] for k, (lo, hi) in pairs.items()}

    return {
        "format": LIGHT_FORMAT,
        "features": dict(bundle["cfg"]["features"]),
        "feature_order": order,
        "scaler": {"mean": [float(x) for x in scaler.mean_], "scale": [float(x) for x in scaler.scale_]},
        "logreg": {"coef": [float(x) for x in logreg.coef_[0]], "intercept": float(logreg.intercept_[0]),
                   "classes": [int(c) for c in logreg.classes_]},
        "ranges": {"inputs": floats(bundle["ranges"]["inputs"]), "features": floats(bundle["ranges"]["features"])},
        "quantiles": {f: [float(x) for x in q] for f, q in bundle["quantiles"].items()},
        "meta": bundle["meta"],
    }


def to_sklearn_bundle(light: dict) -> dict:
    """A full bundle with a REAL scikit-learn Pipeline, rebuilt from the exported numbers (no pickle).

    For places that have scikit-learn but not the version that wrote the .joblib: the browser has
    scikit-learn 1.8 (Pyodide), the model was saved with 1.9, and a pickle is only safe with the same
    version. Setting the fitted attributes by hand gives the same model in any version, and then the
    normal serving code (api/bundle.predict_row: pandas build_features + predict_proba) runs on it.
    Tests: tests/test_light_model.py (same predictions as the original bundle)."""
    import numpy as np  # imported here: the rest of this file needs no numpy
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Step 1: the scaler, exactly as fit() would have left it
    names = np.array(light["feature_order"], dtype=object)
    scaler = StandardScaler()
    scaler.mean_ = np.array(light["scaler"]["mean"])
    scaler.scale_ = np.array(light["scaler"]["scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_, scaler.feature_names_in_ = len(names), names
    scaler.n_samples_seen_ = light["meta"]["n_train"]

    # Step 2: the logistic regression (coefficients, intercept, class labels)
    logreg = LogisticRegression()
    logreg.coef_ = np.array([light["logreg"]["coef"]])
    logreg.intercept_ = np.array([light["logreg"]["intercept"]])
    logreg.classes_ = np.array(light["logreg"].get("classes", [0, 1]))
    logreg.n_features_in_ = len(names)   # no feature names: inside a Pipeline it receives the scaler's array

    # Step 3: the same dict shape as api/bundle.build_bundle (no "format" key -> served as a full bundle)
    return {"model": Pipeline([("scaler", scaler), ("logreg", logreg)]),
            "cfg": {"features": dict(light["features"])},
            "ranges": light["ranges"], "quantiles": light["quantiles"], "meta": light["meta"]}


def build_features_light(cfg_features: dict, inputs: dict[str, float], age: float, sex: str) -> dict[str, float]:
    """The same features as features.build_features, for one patient, without pandas."""
    values = {}
    for b in cfg_features["biomarkers"]:
        v = inputs[b]
        if cfg_features["normalize_by_creatinine"]:
            v = v / inputs["creatinine"]
        if cfg_features["log_transform"]:
            v = math.log1p(v)               # log1p(marker / creatinine), in that order
        values[b] = v
    for d in cfg_features["demographics"]:
        if d == "age":
            values["age"] = float(age)
        elif d == "sex":
            if sex not in ("M", "F"):
                raise ValueError(f"sex must be 'M' or 'F', got {sex!r}")
            values["sex"] = 1.0 if sex == "M" else 0.0
        else:
            raise ValueError(f"Unknown demographic feature {d!r}")
    return values


def _sigmoid(x: float) -> float:
    """1 / (1 + e^-x) without overflow for large |x| (a unit mistake can give log-odds of +-100s)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def predict_row_light(light: dict, inputs: dict[str, float], age: float, sex: str) -> dict:
    """Same output as api/bundle.predict_row (inputs in dataset units), from the exported numbers."""
    if light.get("format") != LIGHT_FORMAT:
        raise ValueError(f"Not a light model ({light.get('format')!r}); export it with export_light()")

    # Step 1: features -> standardised values -> log-odds -> probability
    x = build_features_light(light["features"], inputs, age, sex)
    order, coef = light["feature_order"], light["logreg"]["coef"]
    z = [(x[f] - m) / s for f, m, s in zip(order, light["scaler"]["mean"], light["scaler"]["scale"], strict=True)]
    intercept = light["logreg"]["intercept"]
    probability = _sigmoid(intercept + sum(c * zz for c, zz in zip(coef, z, strict=True)))

    # Step 2: exact contributions (coefficient x standardised value), biggest first
    contributions = [
        {"feature": f, "model_value": x[f], "contribution_log_odds": c * zz,
         "training_percentile": None if f == "sex" else percentile(light["quantiles"][f], x[f])}
        for f, c, zz in zip(order, coef, z, strict=True)
    ]
    contributions.sort(key=lambda d: abs(d["contribution_log_odds"]), reverse=True)

    # Step 3: the shared range / unit checks
    out_of_range, warnings_ = check_ranges(light["ranges"], inputs, age, x)
    return {"probability": probability, "base_log_odds": intercept, "contributions": contributions,
            "out_of_range": out_of_range, "warnings": warnings_}
