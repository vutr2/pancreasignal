"""Phase 5 — serum miRNA data (GSE59856): parsing, tasks, features, model, end-to-end."""
import gzip

import numpy as np
import pandas as pd
import pytest

from pancreasignal.data.geo import _clean_stage, load_mirna, read_series_matrix, select_mirna_task
from pancreasignal.features import build_mirna_features
from pancreasignal.models import MinDetectionRate, build_elasticnet

TINY_MATRIX = "\n".join([
    '!Series_title\t"tiny"',
    '!Sample_title\t"pancreatic cancer P01"\t"healthy control S01"\t"healthy control E01"',
    '!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"',
    '!Sample_characteristics_ch1\t"disease state: pancreatic cancer"\t"disease state: healthy control"\t"disease state: healthy control"',
    '!Sample_characteristics_ch1\t"gender: Male"\t"gender: Female"\t"gender: Male"',
    '!Sample_characteristics_ch1\t"tumor stage: pStage IIB"\t"tumor stage: uncertain"\t"tumor stage: uncertain"',
    '!Sample_characteristics_ch1\t"age: 70"\t"age: uncertain"\t"age: 55"',
    '!Sample_characteristics_ch1\t"operability: operable"\t"operability: uncertain"\t"operability: uncertain"',
    "!series_matrix_table_begin",
    '"ID_REF"\t"GSM2"\t"GSM1"\t"GSM3"',          # columns NOT in sample order on purpose
    '"MIMAT0000001"\t2.612\t9.5\t3.0',
] + [f'"MIMAT{i:07d}"\t5.0\t6.0\t7.0' for i in range(2, 102)] + ["!series_matrix_table_end", ""])


@pytest.fixture
def tiny_matrix(tmp_path):
    path = tmp_path / "GSE0_series_matrix.txt.gz"
    with gzip.open(path, "wt") as f:
        f.write(TINY_MATRIX)
    return path


def test_read_series_matrix(tiny_matrix):
    meta, expr = read_series_matrix(tiny_matrix)
    assert meta["geo_accession"] == ["GSM1", "GSM2", "GSM3"]
    assert meta["disease state"][0] == "pancreatic cancer"
    assert expr.shape == (3, 101)


def test_load_mirna_matches_samples_by_id(tiny_matrix):
    df = load_mirna(tiny_matrix)
    assert list(df["diagnosis"]) == ["pdac", "control", "control"]
    assert df.loc[0, "MIMAT0000001"] == 9.5, "GSM1's value, even though GSM1 is the 2nd table column"
    assert np.isnan(df.loc[1, "age"]), "'uncertain' age must become NaN"
    assert list(df["sex"]) == ["M", "F", "M"]
    assert df.loc[0, "stage"] == "IIB" and pd.isna(df.loc[1, "stage"]), "pandas stores None as NaN"
    assert list(df["title_prefix"]) == ["P", "S", "E"]


def test_mixed_characteristics_line_fails(tmp_path):
    bad = TINY_MATRIX.replace('"gender: Female"', '"age: 40"')
    path = tmp_path / "bad_series_matrix.txt.gz"
    with gzip.open(path, "wt") as f:
        f.write(bad)
    with pytest.raises(ValueError, match="mixes fields"):
        read_series_matrix(path)


def test_clean_stage():
    assert [_clean_stage(s) for s in ["cStage IV", "pStage IIA", "ypStage IB", "uncertain"]] == ["IV", "IIA", "IB", None]


def test_select_mirna_task(fake_mirna):
    rows, y = select_mirna_task(fake_mirna, "pdac_vs_biliary")
    assert set(rows["diagnosis"]) == {"pdac", "biliary"}
    assert (y == (rows["diagnosis"] == "pdac")).all()


def test_build_mirna_features(fake_mirna, mirna_cfg):
    X = build_mirna_features(fake_mirna, mirna_cfg)
    assert X.shape[1] == 40 and all(c.startswith("MIMAT") for c in X.columns)
    mirna_cfg["features"].update(mirna=False, demographics=["age", "sex"])
    X = build_mirna_features(fake_mirna, mirna_cfg)
    assert list(X.columns) == ["age", "sex"] and set(X["sex"]) <= {0, 1}


def test_detection_filter_uses_training_rows_only():
    X = np.array([[2.612, 5.0], [2.612, 5.0], [9.0, 5.0], [9.0, 5.0]])
    # fitted on the first two rows only: column 0 is never detected there -> dropped
    sel = MinDetectionRate(floor=2.612, min_rate=0.5).fit(X[:2])
    assert list(sel.get_support()) == [False, True]
    assert list(MinDetectionRate(min_rate=0.0).fit(X).get_support()) == [True, True]


def test_elasticnet_finds_signal(fake_mirna, mirna_cfg):
    rows, y = select_mirna_task(fake_mirna, "pdac_vs_control")
    X = build_mirna_features(rows, mirna_cfg)
    model = build_elasticnet(mirna_cfg["model"]["params"]["elasticnet"]).fit(X, y)
    kept = model[:-1].get_feature_names_out()
    assert "MIMAT0000030" not in kept, "mostly-undetected miRNAs are filtered out"
    coef = model.named_steps["clf"].coef_[0]
    signal = [i for i, name in enumerate(kept) if name in {f"MIMAT{i:07d}" for i in range(5)}]
    assert np.count_nonzero(coef[signal]) >= 3, "the 5 planted signal miRNAs should mostly be kept"


def test_end_to_end_mirna(fake_mirna, mirna_cfg, monkeypatch):
    from pancreasignal import train

    monkeypatch.setattr(train, "load_mirna", lambda path: fake_mirna)
    res = train.run(mirna_cfg)["results"]
    assert res["dataset"] == "mirna" and res["split"] == "stratified_kfold"
    assert 0.5 < res["model_all"]["auroc"] <= 1.0
    assert "ca199_baseline" not in res, "no CA19-9 in the miRNA data"
    assert all("n_nonzero_coefs" in f for f in res["per_fold"].values())
