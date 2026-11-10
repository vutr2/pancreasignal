"""Phase 6 — model bundle + FastAPI app. Uses the fake urine table, never the real model file."""
import copy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pancreasignal.api.app import create_app
from pancreasignal.api.bundle import build_bundle, load_bundle, predict_row, save_bundle


@pytest.fixture
def bundle(fake_urine, cfg, monkeypatch):
    from pancreasignal import train

    monkeypatch.setattr(train, "load_urine", lambda path: fake_urine)
    return build_bundle(cfg)


@pytest.fixture
def client(bundle):
    with TestClient(create_app(bundle=bundle)) as c:
        yield c


@pytest.fixture
def patient():
    """A typical patient of the fake table, in dataset units."""
    return {"lyve1": {"value": 1.5, "unit": "ng/mL"}, "reg1b": {"value": 30.0, "unit": "ng/mL"},
            "tff1": {"value": 200.0, "unit": "ng/mL"}, "creatinine": {"value": 1.2, "unit": "mg/mL"},
            "age": 60, "sex": "M"}


def test_predict_ok(client, patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert len(body["reasons"]) == 3
    assert body["out_of_distribution"] is False
    assert any("Not for clinical use" in w for w in body["warnings"])


def test_reasons_add_up_to_the_prediction(bundle):
    # exact explanation: intercept + sum of contributions = log-odds of the probability
    r = predict_row(bundle, {"LYVE1": 1.5, "REG1B": 30.0, "TFF1": 200.0, "creatinine": 1.2}, 60, "M")
    log_odds = r["base_log_odds"] + sum(c["contribution_log_odds"] for c in r["contributions"])
    assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(r["probability"])


def test_reasons_match_the_shap_library(bundle):
    import pandas as pd
    import shap

    model = bundle["model"]
    X = pd.DataFrame([[np.log1p(1.5 / 1.2), np.log1p(30.0 / 1.2), np.log1p(200.0 / 1.2), 60.0, 1.0]],
                     columns=bundle["meta"]["features"])
    z = model.named_steps["scaler"].transform(X)
    # background = the training data in standardised space, whose mean is 0
    background = np.zeros((1, z.shape[1]))
    clf = model.named_steps["logreg"]
    expected = np.ravel(shap.LinearExplainer((clf.coef_[0], clf.intercept_[0]), background).shap_values(z))
    ours = {c["feature"]: c["contribution_log_odds"]
            for c in predict_row(bundle, {"LYVE1": 1.5, "REG1B": 30.0, "TFF1": 200.0, "creatinine": 1.2}, 60, "M")["contributions"]}
    assert [ours[f] for f in ["LYVE1", "REG1B", "TFF1", "age", "sex"]] == pytest.approx(list(expected))


def test_same_patient_in_other_units_gives_the_same_probability(client, patient):
    p_dataset_units = client.post("/predict", json=patient).json()["probability"]
    other = copy.deepcopy(patient)
    other["creatinine"] = {"value": 1.2 / 0.11312, "unit": "mmol/L"}
    other["lyve1"] = {"value": 1500.0, "unit": "pg/mL"}
    other["tff1"] = {"value": 200.0, "unit": "µg/L"}
    assert client.post("/predict", json=other).json()["probability"] == pytest.approx(p_dataset_units)


def test_unit_mistake_is_flagged(client, patient):
    # value written in ng/mL but labelled µg/mL -> 1000x too high -> out of range + unit warning
    patient["lyve1"] = {"value": 1.5, "unit": "µg/mL"}
    body = client.post("/predict", json=patient).json()
    assert body["out_of_distribution"] is True
    assert any("check the unit" in w for w in body["warnings"])


@pytest.mark.parametrize("change", [
    {"lyve1": {"value": 1.0, "unit": "ng/dL"}},       # unknown unit
    {"creatinine": {"value": 0.0, "unit": "mg/mL"}},  # division by zero
    {"tff1": {"value": -3.0, "unit": "ng/mL"}},       # negative
    {"sex": "X"},
    {"age": 300},
    {"lyve_1": {"value": 1.0, "unit": "ng/mL"}},      # typo in a field name
])
def test_invalid_requests_get_422(client, patient, change):
    assert client.post("/predict", json={**patient, **change}).status_code == 422


def test_unknown_unit_error_lists_allowed_units(client, patient):
    patient["creatinine"] = {"value": 1.0, "unit": "mg/day"}
    assert "mmol/L" in client.post("/predict", json=patient).text


def test_sex_spellings(client, patient):
    for spelling in ["male", "Male", "m"]:
        assert client.post("/predict", json={**patient, "sex": spelling}).status_code == 200


def test_missing_field_gets_422(client, patient):
    del patient["creatinine"]
    assert client.post("/predict", json=patient).status_code == 422


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model").json()
    assert info["task"] == "pdac_vs_rest" and info["dataset_units"]["creatinine"] == "mg/mL"


def test_saved_bundle_roundtrip_and_tamper_check(bundle, tmp_path):
    path = tmp_path / "m.joblib"
    save_bundle(bundle, path)
    assert load_bundle(path)["meta"]["task"] == "pdac_vs_rest"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_bundle(path)
