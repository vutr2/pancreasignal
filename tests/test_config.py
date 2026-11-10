"""Phase 4 — config inheritance (`base:`)."""
from pancreasignal.config import load_config


def test_load_config_base(tmp_path):
    (tmp_path / "parent.yaml").write_text(
        "name: parent\nfeatures: {biomarkers: [A, B], log_transform: true}\nevaluation: {n_bootstrap: 10}\n"
    )
    (tmp_path / "child.yaml").write_text("base: parent.yaml\nname: child\nfeatures: {biomarkers: []}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["name"] == "child"
    assert cfg["features"] == {"biomarkers": [], "log_transform": True}, "nested dicts merge key by key"
    assert cfg["evaluation"] == {"n_bootstrap": 10}, "inherited unchanged"
    assert "base" not in cfg


def test_real_age_only_config():
    cfg = load_config("configs/urine_age_only.yaml")
    main = load_config("configs/urine.yaml")
    assert cfg["features"]["demographics"] == ["age"]
    assert cfg["evaluation"] == main["evaluation"], "baseline must use the same evaluation settings"
