"""Phase 4 — reading runs back from reports/."""
import json

from pancreasignal.reporting import get_run, load_runs, variant_of


def test_variant_of():
    assert variant_of({"model": "logreg"}) == "logreg", "old reports have no name"
    assert variant_of({"model": "logreg", "name": "urine_panel", "calibrated": True}) == "logreg_calibrated"
    assert variant_of({"model": "logreg", "name": "age_only"}) == "age_only"


def test_load_runs_keeps_newest(tmp_path):
    for run_id, auroc in [("20260101-000000_t_logreg", 0.5), ("20260102-000000_t_logreg", 0.9)]:
        (tmp_path / f"{run_id}_metrics.json").write_text(json.dumps(
            {"task": "t", "model": "logreg", "split": "x", "model_all": {"auroc": auroc}}))
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert get_run(runs, "t", "logreg")["results"]["model_all"]["auroc"] == 0.9
