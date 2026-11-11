"""Phase 9 — the agent: clinical units, critic rules, graph with pause/resume, reports, traces.
No network: fake risk client and fake RAG; the real model is used only if it was exported locally."""
import json

import pytest
import yaml
from pydantic import ValidationError

from pancreasignal.agents.clinical import CaseInput, ClinicalContext
from pancreasignal.agents.critic import Q_JAUNDICE, review_case
from pancreasignal.agents.graph import AgentDeps, build_graph, run_case
from pancreasignal.agents.trace import Tracer
from pancreasignal.config import PROJECT_ROOT

URINE = {"lyve1": {"value": 1.0, "unit": "ng/mL"}, "reg1b": {"value": 30, "unit": "ng/mL"},
         "tff1": {"value": 200, "unit": "ng/mL"}, "creatinine": {"value": 0.8, "unit": "mg/mL"}, "age": 60, "sex": "F"}


def risk_out(p=0.3, creatinine=0.8, warnings=(), ood=False):
    """A /predict-shaped response with only what the critic and report read."""
    return {"probability": p, "warnings": list(warnings) + ["Research prototype. Not for clinical use."],
            "out_of_distribution": ood, "out_of_range": [{"feature": "LYVE1"}] if ood else [],
            "inputs_in_dataset_units": {"creatinine": creatinine},
            "reasons": [{"feature": "LYVE1", "direction": "raises", "training_percentile": 80.0,
                         "text": "LYVE1 per creatinine is at percentile 80 of training samples; it raises the risk."}],
            "model": {"task": "pdac_vs_rest", "performance_reference": {"auroc": 0.785}}}


# ---------------------------------------------------------------- clinical input
def test_clinical_units():
    c = ClinicalContext(total_bilirubin={"value": 171, "unit": "µmol/L"}, plasma_ca19_9={"value": 40, "unit": "kU/L"})
    n = c.normalised()
    assert n["bilirubin_mg_dl"] == pytest.approx(10.0) and n["ca19_9_u_ml"] == 40
    with pytest.raises(ValidationError, match="Unknown unit"):
        ClinicalContext(total_bilirubin={"value": 1, "unit": "g/L"})


def test_case_id_must_be_safe_as_a_file_name():
    with pytest.raises(ValidationError):
        CaseInput(case_id="../../etc/passwd", urine=URINE)


# ---------------------------------------------------------------- critic rules
@pytest.mark.parametrize("kwargs,decision,codes", [
    ({"risk": risk_out()}, "proceed", []),
    ({"risk": risk_out(warnings=["LYVE1 = 5600 ... check the unit."], ood=True)}, "abstain", ["UNIT_SUSPECT"]),
    ({"risk": risk_out(ood=True)}, "proceed", ["OUT_OF_DISTRIBUTION"]),
    ({"risk": risk_out(creatinine=0.1)}, "proceed", ["DILUTE_URINE"]),
    ({"risk": risk_out(creatinine=3.5)}, "proceed", ["CONCENTRATED_URINE"]),
    ({"risk": risk_out(), "clinical": {"jaundice": True, "ca19_9_u_ml": 900}}, "proceed", ["CHOLESTASIS_CA199"]),
    ({"risk": risk_out(), "clinical": {"bilirubin_mg_dl": 5.0}}, "proceed", ["CHOLESTASIS"]),
    ({"risk": risk_out(), "clinical": {"bilirubin_mg_dl": 2.0, "ca19_9_u_ml": 37}}, "proceed", []),   # both at the limit
    ({"risk": risk_out(), "clinical": {"ca19_9_u_ml": 1}}, "proceed", ["CA199_VERY_LOW"]),
    ({"risk": risk_out(), "clinical": {"new_onset_diabetes": True}}, "proceed", ["NEW_ONSET_DIABETES"]),
    ({"risk": None, "risk_error": "ConnectError: API down"}, "abstain", ["RISK_UNAVAILABLE"]),
    ({"intake_errors": ["urine.creatinine: Field required"], "risk": None}, "abstain", ["INTAKE_INVALID"]),
])
def test_critic_rules(kwargs, decision, codes):
    out = review_case(kwargs.get("intake_errors", []), kwargs.get("risk"), kwargs.get("clinical"),
                      risk_error=kwargs.get("risk_error"))
    assert out["decision"] == decision and [f["code"] for f in out["findings"]] == codes
    assert all(f["message_vi"] for f in out["findings"]), "every finding is bilingual"


def test_jaundice_finding_asks_the_rag():
    out = review_case([], risk_out(), {"jaundice": True, "ca19_9_u_ml": 900})
    assert out["findings"][0]["rag_question"] == Q_JAUNDICE


# ---------------------------------------------------------------- the graph
class FakeRisk:
    def __init__(self, output=None, error=None):
        self.output, self.error, self.calls = output or risk_out(), error, 0
        self.name = "fake"

    def predict(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output


def fake_rag(question):
    return {"answer": f"Evidence about: {question} [PMC1#000]", "refused": False,
            "citations": [{"chunk_id": "PMC1#000", "title": "T", "journal": "J", "year": 2020, "section": "S"}]}


def run(tmp_path, raw, risk=None, rag=fake_rag, reply=None):
    tracer = Tracer("run1", tmp_path / "traces")
    app = build_graph(AgentDeps(risk_client=risk or FakeRisk(), rag_fn=rag, tracer=tracer, report_dir=str(tmp_path / "agent")))
    reply = reply or {"decision": "approve", "reviewer": "tester", "note": "ok"}
    return run_case(app, raw, "thread-1", reviewer=lambda draft: reply), tracer


def test_graph_pauses_for_review_then_reports(tmp_path):
    raw = {"case_id": "c1", "urine": URINE, "clinical": {"jaundice": True, "plasma_ca19_9": {"value": 900, "unit": "U/mL"}}}
    tracer = Tracer("run1", tmp_path / "traces")
    app = build_graph(AgentDeps(risk_client=FakeRisk(), rag_fn=fake_rag, tracer=tracer, report_dir=str(tmp_path / "agent")))
    config = {"configurable": {"thread_id": "t"}}
    out = app.invoke({"raw": raw}, config)
    draft = out["__interrupt__"][0].value
    assert draft["decision"] == "proceed" and draft["probability"] == 0.3, "paused with a draft, nothing written yet"
    assert not (tmp_path / "agent").exists()
    from langgraph.types import Command
    final = app.invoke(Command(resume={"decision": "approve", "reviewer": "dr-x"}), config)
    md = (tmp_path / "agent" / "run1.md").read_text()
    assert "## English" in md and "## Tiếng Việt" in md and "Model output: 30%" in md and "Kết quả của model: 30%" in md
    assert "Evidence about: Is CA19-9 reliable" in md and "ĐÃ DUYỆT" in md
    assert final["review"]["reviewer"] == "dr-x"
    nodes = [(e["node"], e["status"]) for e in tracer.events]
    assert ("human_review", "paused") in nodes and nodes[-1] == ("report", "ok")
    assert len((tmp_path / "traces" / "run1.jsonl").read_text().splitlines()) == len(tracer.events)


def test_rejected_case_is_still_reported(tmp_path):
    _final, _ = run(tmp_path, {"case_id": "c1", "urine": URINE}, reply={"decision": "reject", "reviewer": "dr-y", "note": "repeat test"})
    summary = json.loads((tmp_path / "agent" / "run1.json").read_text())
    assert summary["review"] == "reject" and "BỊ TỪ CHỐI" in (tmp_path / "agent" / "run1.md").read_text()


def test_invalid_case_abstains_without_calling_the_model(tmp_path):
    risk = FakeRisk()
    urine = {k: v for k, v in URINE.items() if k != "creatinine"}
    final, _ = run(tmp_path, {"case_id": "c7", "urine": urine}, risk=risk)
    assert final["critic"]["decision"] == "abstain" and risk.calls == 0
    md = (tmp_path / "agent" / "run1.md").read_text()
    assert "Risk estimate withheld" in md and "Model output" not in md, "no number is shown when abstaining"


def test_model_failure_becomes_a_block(tmp_path):
    final, _ = run(tmp_path, {"case_id": "c1", "urine": URINE}, risk=FakeRisk(error=ConnectionError("API down")))
    assert [f["code"] for f in final["critic"]["findings"]] == ["RISK_UNAVAILABLE"]


def test_rag_failure_does_not_stop_the_run(tmp_path):
    def broken_rag(q):
        raise RuntimeError("NVIDIA_API_KEY is not set")
    final, _ = run(tmp_path, {"case_id": "c1", "urine": URINE, "clinical": {"new_onset_diabetes": True}}, rag=broken_rag)
    assert "NVIDIA_API_KEY" in final["evidence"][0]["error"]
    assert (tmp_path / "agent" / "run1.md").exists()


def test_bad_review_reply_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Review reply"):
        run(tmp_path, {"case_id": "c1", "urine": URINE}, reply={"decision": "maybe"})


# ---------------------------------------------------------------- risk clients
def test_http_and_local_risk_agree(fake_urine, cfg, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient

    from pancreasignal import train
    from pancreasignal.agents.risk import HttpRisk, LocalRisk
    from pancreasignal.api.app import create_app
    from pancreasignal.api.bundle import build_bundle
    from pancreasignal.api.schemas import PredictRequest

    monkeypatch.setattr(train, "load_urine", lambda path: fake_urine)
    bundle = build_bundle(cfg)
    with TestClient(create_app(bundle=bundle)) as client:
        monkeypatch.setattr(httpx, "post", lambda url, json, timeout: client.post("/predict", json=json))
        request = PredictRequest.model_validate(URINE)
        assert HttpRisk("http://test").predict(request)["probability"] == pytest.approx(
            LocalRisk(bundle=bundle).predict(request)["probability"])


MODEL = PROJECT_ROOT / "models" / "urine_v1.joblib"


@pytest.mark.skipif(not MODEL.exists(), reason="real model not exported (python scripts/export_model.py)")
def test_demo_cases_with_the_real_model(tmp_path):
    """Regression: all evals/agent_cases.yaml cases get their expected critic decision."""
    from pancreasignal.agents.risk import LocalRisk

    risk = LocalRisk()
    for c in yaml.safe_load((PROJECT_ROOT / "evals" / "agent_cases.yaml").read_text())["cases"]:
        raw = {k: c[k] for k in ("case_id", "urine", "clinical") if k in c}
        final, _ = run(tmp_path / c["case_id"], raw, risk=risk, rag=None)
        got = (final["critic"]["decision"], sorted(f["code"] for f in final["critic"]["findings"]))
        assert got == (c["expect"]["decision"], sorted(c["expect"]["findings"])), c["case_id"]
