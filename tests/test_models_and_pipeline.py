"""Phase 2-4 — the last test is the "everything works" check."""
import numpy as np

from pancreasignal.data.urine import select_task
from pancreasignal.features import build_features
from pancreasignal.models import build_logreg, build_xgboost


def _xy(fake_urine, cfg):
    rows, y = select_task(fake_urine, "pdac_vs_rest")
    return build_features(rows, cfg), y


def test_logreg(fake_urine, cfg):
    X, y = _xy(fake_urine, cfg)
    p = build_logreg({"C": 1.0, "max_iter": 1000}).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1))


def test_xgboost(fake_urine, cfg):
    X, y = _xy(fake_urine, cfg)
    p = build_xgboost({"n_estimators": 20, "max_depth": 2}).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(y),)


def test_end_to_end(fake_urine, cfg, tmp_path, monkeypatch):
    """Runs the full train.run() on the fake table. Passes when Phases 1-3 are done."""
    from pancreasignal import train

    monkeypatch.setattr(train, "load_urine", lambda path: fake_urine)
    out = train.run(cfg)
    res = out["results"]
    assert set(res["per_fold"]) == {"AAA", "BBB", "CCC"}
    assert 0.5 < res["model_all"]["auroc"] <= 1.0
    assert "ca199_baseline" in res


def test_calibrated_logreg(fake_urine, cfg):
    from pancreasignal.models import build_model

    cfg["model"]["calibrate"] = True
    X, y = _xy(fake_urine, cfg)
    p = build_model(cfg).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1))


def test_end_to_end_reports_stage_and_ece(fake_urine, cfg, monkeypatch):
    from pancreasignal import train

    monkeypatch.setattr(train, "load_urine", lambda path: fake_urine)
    res = train.run(cfg)["results"]
    assert res["model_all"]["ece"] is not None
    assert res["sens_by_stage"]["spec95"]["I-II"]["n_pdac"] > 0
