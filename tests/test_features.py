"""Phase 2 — fail until you implement src/pancreasignal/features.py."""
import numpy as np
import pytest

from pancreasignal.features import (
    build_features,
    encode_sex,
    log_transform,
    normalize_by_creatinine,
)

BIOS = ["LYVE1", "REG1B", "TFF1"]


def test_normalize_by_creatinine(fake_urine):
    before = fake_urine.copy()
    out = normalize_by_creatinine(fake_urine, BIOS)
    assert np.allclose(out["LYVE1"], fake_urine["LYVE1"] / fake_urine["creatinine"])
    assert out["age"].equals(fake_urine["age"])
    assert fake_urine.equals(before), "must not modify the input"


def test_log_transform(fake_urine):
    out = log_transform(fake_urine, ["TFF1"])
    assert np.allclose(out["TFF1"], np.log1p(fake_urine["TFF1"]))


def test_encode_sex(fake_urine):
    out = encode_sex(fake_urine)
    assert set(out["sex"].unique()) <= {0, 1}
    assert (out["sex"] == (fake_urine["sex"] == "M").astype(int)).all()


def test_build_features(fake_urine, cfg):
    X = build_features(fake_urine, cfg)
    assert list(X.columns) == BIOS + ["age", "sex"]
    assert len(X) == len(fake_urine)
    assert not X.isna().any().any()
    assert "diagnosis" not in X.columns, "label leaked into features!"


def test_log_transform_only_given_columns(fake_urine):
    out = log_transform(fake_urine, ["TFF1"])
    assert out["age"].equals(fake_urine["age"]), "age must not be log-transformed"
    assert out["creatinine"].equals(fake_urine["creatinine"])


def test_encode_sex_rejects_unknown(fake_urine):
    df = fake_urine.copy()
    df.loc[0, "sex"] = "m"
    with pytest.raises(ValueError, match="sex must be"):
        encode_sex(df)


def test_build_features_values(fake_urine, cfg):
    """Checks VALUES, not just column names: the transforms must really be applied."""
    X = build_features(fake_urine, cfg)
    for b in BIOS:
        assert np.allclose(X[b], np.log1p(fake_urine[b] / fake_urine["creatinine"])), b
    assert X["age"].equals(fake_urine["age"]), "age must stay raw"


def test_build_features_flags_off(fake_urine, cfg):
    cfg["features"]["normalize_by_creatinine"] = False
    cfg["features"]["log_transform"] = False
    X = build_features(fake_urine, cfg)
    assert np.allclose(X["LYVE1"], fake_urine["LYVE1"])


def test_build_features_age_only(fake_urine, cfg):
    # configs/urine_age_only.yaml: no biomarkers, no sex
    cfg["features"].update(biomarkers=[], demographics=["age"],
                           normalize_by_creatinine=False, log_transform=False)
    X = build_features(fake_urine, cfg)
    assert list(X.columns) == ["age"]
    assert X["age"].equals(fake_urine["age"])
