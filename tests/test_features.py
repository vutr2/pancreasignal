"""Phase 2 — fail until you implement src/pancreasignal/features.py."""
import numpy as np

from pancreasignal.features import build_features, encode_sex, log_transform, normalize_by_creatinine

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
