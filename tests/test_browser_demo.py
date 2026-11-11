"""browser_demo.py: the Python side of the free static demo, run here in CPython (the same code runs in
Pyodide). The 8 synthetic cases must get the same critic decisions as the real agent."""
import json

import pytest
import yaml

from pancreasignal import browser_demo
from pancreasignal.agents import critic
from pancreasignal.api.light import LIGHT_FORMAT, export_light
from pancreasignal.config import PROJECT_ROOT

CASES = yaml.safe_load((PROJECT_ROOT / "evals" / "agent_cases.yaml").read_text())["cases"]
MODEL = PROJECT_ROOT / "models" / "urine_v1.joblib"
SNAPSHOT = {"meta": {}, "answers": {q: {"question": q, "answer": f"About {q} [PMC1#000]", "refused": False,
                                        "citations": [{"chunk_id": "PMC1#000", "pmcid": "PMC1", "title": "T",
                                                       "journal": "J", "year": 2020, "section": "S"}]}
                                    for q in critic.RAG_QUESTIONS}}


def fake_light() -> dict:
    """A tiny hand-made light model (no model file needed): every marker raises the risk."""
    feats = ["LYVE1", "REG1B", "TFF1", "age", "sex"]
    return {"format": LIGHT_FORMAT,
            "features": {"biomarkers": feats[:3], "demographics": ["age", "sex"],
                         "normalize_by_creatinine": True, "log_transform": True},
            "feature_order": feats, "scaler": {"mean": [1, 3, 5, 60, 0.5], "scale": [1, 1, 1, 10, 0.5]},
            "logreg": {"coef": [1, 0.5, 0.5, 0.2, 0.1], "intercept": -1.0},
            "ranges": {"inputs": {"LYVE1": [0, 24], "REG1B": [0, 1400], "TFF1": [0, 13000],
                                  "creatinine": [0.05, 4.1], "age": [26, 89]},
                       "features": {f: [0, 12] for f in feats[:3]} | {"age": [26, 89], "sex": [0, 1]}},
            "quantiles": {f: [i / 10 for i in range(101)] for f in feats},
            "meta": {"name": "fake", "task": "pdac_vs_rest", "n_train": 10, "trained_at": "2026-11-11",
                     "performance_reference": {"auroc": 0.5}}}


def _setup(light: dict) -> dict:
    return json.loads(browser_demo.setup(json.dumps(light), json.dumps(SNAPSHOT), json.dumps(CASES)))


def test_draft_before_setup_is_an_error(monkeypatch):
    monkeypatch.setitem(browser_demo._session, "engines", {})
    with pytest.raises(RuntimeError, match="setup"):
        browser_demo.draft("[]")


def test_sklearn_engine_is_checked_against_the_python_engine():
    """enable_sklearn() rebuilds a real Pipeline and compares it with the pure-Python copy on the cases."""
    _setup(fake_light())
    with pytest.raises(ValueError, match="not loaded"):
        browser_demo.set_engine("sklearn")
    check = json.loads(browser_demo.enable_sklearn())
    assert check["cases_checked"] == 7 and check["max_probability_difference"] < 1e-12
    assert browser_demo.set_engine("sklearn") == "numpy + scikit-learn"
    form = json.loads(browser_demo.setup(json.dumps(fake_light()), json.dumps(SNAPSHOT), json.dumps(CASES)))["cases"][0]["form"]
    browser_demo.enable_sklearn()
    browser_demo.set_engine("sklearn")
    assert json.loads(browser_demo.draft(json.dumps(form)))["engine"] == "numpy + scikit-learn"


def test_setup_gives_the_page_its_form():
    info = _setup(fake_light())
    assert [c["case_id"] for c in info["cases"]] == [c["case_id"] for c in CASES]
    assert len(info["cases"][0]["form"]) == 17 and "µg/mL" in info["units"]["protein"]


def test_draft_pauses_then_finish_reports_once():
    info = _setup(fake_light())
    form = next(c["form"] for c in info["cases"] if c["case_id"] == "c03_jaundice_high_ca199")
    d = json.loads(browser_demo.draft(json.dumps(form)))
    assert d["decision"] == "proceed" and "NOT REVIEWED" in d["markdown"]
    assert [t["node"] for t in d["timings"]] == ["intake", "risk", "critic", "rag", "human_review"]
    assert "Is CA19-9 reliable" in d["markdown"]                     # evidence from the snapshot
    f = json.loads(browser_demo.finish("reject", "tester", "not convinced"))
    assert "REJECTED" in f["status"] and "tester: not convinced" in f["markdown"]
    with pytest.raises(RuntimeError, match="already reviewed"):
        browser_demo.finish("approve")


@pytest.mark.skipif(not MODEL.exists(), reason="models/urine_v1.joblib not exported")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_demo_cases_in_the_browser_path(case):
    """Real model (light copy) + the browser path = the expected critic decision for every case."""
    from pancreasignal.api.bundle import load_bundle

    info = _setup(json.loads(json.dumps(export_light(load_bundle(MODEL)))))
    browser_demo.enable_sklearn()
    form = next(c["form"] for c in info["cases"] if c["case_id"] == case["case_id"])
    for engine in ("python", "sklearn"):
        browser_demo.set_engine(engine)
        d = json.loads(browser_demo.draft(json.dumps(form)))
        assert d["decision"] == case["expect"]["decision"], engine
        assert sorted(d["findings"]) == sorted(case["expect"]["findings"]), engine
