"""Web demo (src/pancreasignal/demo.py) and the Space build (scripts/build_space.py).
No network, no browser: the page is built but not launched; the handlers are called directly."""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

gr = pytest.importorskip("gradio")

from pancreasignal.agents import critic
from pancreasignal.agents.clinical import CaseInput
from pancreasignal.api.units import normalize_unit
from pancreasignal.config import PROJECT_ROOT
from pancreasignal.demo import (
    DailyBudget,
    DemoAgent,
    answer_markdown,
    build_ui,
    case_from_form,
    form_from_case,
    headline,
    results_markdown,
    snapshot_rag_fn,
)

CASES = yaml.safe_load((PROJECT_ROOT / "evals" / "agent_cases.yaml").read_text())["cases"]
BY_ID = {c["case_id"]: c for c in CASES}
MODEL = PROJECT_ROOT / "models" / "urine_v1.joblib"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="models/urine_v1.joblib not exported")


def fake_snapshot():
    cite = [{"chunk_id": "PMC1#000", "pmcid": "PMC1", "title": "T", "journal": "J", "year": 2020, "section": "S"}]
    return {"meta": {"answer_model": "m", "built_at": "2026-11-11T00:00:00+00:00"},
            "answers": {q: {"question": q, "answer": f"About {q} [PMC1#000]", "refused": False, "citations": cite}
                        for q in critic.RAG_QUESTIONS}}


class FakeRisk:
    def predict(self, request):
        return {"probability": 0.3, "out_of_distribution": False, "out_of_range": [],
                "inputs_in_dataset_units": {"creatinine": 0.8}, "reasons": [], "warnings": ["w"],
                "model": {"task": "pdac_vs_rest", "performance_reference": {"auroc": 0.785}}}


def _units_normalised(case: dict) -> dict:
    """Same case with every unit spelled the normalised way ('µg/mL' and 'ug/mL' compare equal)."""
    def norm(section):
        return {k: ({**v, "unit": normalize_unit(v["unit"])} if isinstance(v, dict) else v)
                for k, v in (section or {}).items()}
    return {"case_id": case["case_id"], "urine": norm(case["urine"]), "clinical": norm(case.get("clinical"))}


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_form_round_trip_keeps_every_case(case):
    """Loading a case into the form and reading it back gives the same case JSON (nothing invented)."""
    raw = case_from_form(*form_from_case(case))
    expected = {k: case[k] for k in ("case_id", "urine", "clinical")}
    assert _units_normalised(raw) == _units_normalised(expected)


def test_empty_fields_are_left_out_for_intake_to_report():
    values = form_from_case(BY_ID["c01_low_risk"])
    values[7] = None                                   # creatinine value cleared
    raw = case_from_form(*values)
    assert "creatinine" not in raw["urine"]
    with pytest.raises(ValueError, match="creatinine"):
        CaseInput.model_validate(raw)


def test_snapshot_covers_every_critic_question():
    """critic.RAG_QUESTIONS must list every Q_ constant, or the demo would miss evidence."""
    assert set(critic.RAG_QUESTIONS) == {v for k, v in vars(critic).items() if k.startswith("Q_")}
    fn = snapshot_rag_fn(fake_snapshot())
    assert fn(critic.Q_LEWIS)["citations"]
    with pytest.raises(KeyError, match="snapshot"):
        fn("a question the critic never asks")


def test_run_pauses_then_approve_writes_the_report(tmp_path):
    agent = DemoAgent(FakeRisk(), snapshot_rag_fn(fake_snapshot()), work_dir=tmp_path)
    started = agent.start(case_from_form(*form_from_case(BY_ID["c03_jaundice_high_ca199"])))
    state = started["state"]
    assert [f["code"] for f in state["critic"]["findings"]] == ["CHOLESTASIS_CA199"]
    assert state["evidence"][0]["question"] == critic.Q_JAUNDICE
    assert [e["status"] for e in started["trace"]][-1] == "paused"
    assert "Model output: 30%" in headline(state)

    final = agent.finish(started["run_id"], "approve", "tester", "looks fine")
    report = Path(final["state"]["report_paths"]["markdown"]).read_text()
    assert "APPROVED" in report and "tester: looks fine" in report
    with pytest.raises(KeyError):                      # a run can be finished once only
        agent.finish(started["run_id"], "reject")


def test_oldest_paused_runs_are_dropped(tmp_path):
    agent = DemoAgent(FakeRisk(), None, work_dir=tmp_path, max_open_runs=2)
    ids = [agent.start({"case_id": f"c{i}", "urine": BY_ID["c01_low_risk"]["urine"]})["run_id"] for i in range(3)]
    with pytest.raises(KeyError):
        agent.finish(ids[0], "approve")
    assert agent.finish(ids[2], "reject")["state"]["review"]["decision"] == "reject"


@needs_model
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_demo_cases_through_the_web_path(case, tmp_path):
    """Form -> real model -> critic gives the expected decision, and every RAG question is in the snapshot."""
    from pancreasignal.agents.risk import LocalRisk

    agent = DemoAgent(LocalRisk(bundle_path=MODEL), snapshot_rag_fn(fake_snapshot()), work_dir=tmp_path)
    state = agent.start(case_from_form(*form_from_case(case)))["state"]
    assert state["critic"]["decision"] == case["expect"]["decision"]
    assert sorted(f["code"] for f in state["critic"]["findings"]) == sorted(case["expect"]["findings"])
    assert all("error" not in e for e in state["evidence"])
    if state["critic"]["decision"] == "abstain":
        assert headline(state).startswith("**Risk estimate withheld**")


def test_daily_budget():
    budget = DailyBudget(2)
    assert [budget.take() for _ in range(3)] == [True, True, False]


def test_answer_markdown_links_sources():
    res = fake_snapshot()["answers"][critic.Q_LEWIS]
    md = answer_markdown({**res, "model_time_s": 1.234})
    assert "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/" in md and "1.2 s" in md


def test_results_block_for_the_web_page(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- RESULTS:START -->\n| a |\n\n![ROC](docs/figures/roc.png)\n"
                      "See [RESULTS.md](RESULTS.md) and [paper](https://example.org).\n<!-- RESULTS:END -->\n")
    text, _ = results_markdown(paths=(readme,), repo_url="https://github.com/u/r")
    assert "![" not in text
    assert "(https://github.com/u/r/blob/main/RESULTS.md)" in text and "(https://example.org)" in text


def test_page_builds_with_and_without_live_questions(tmp_path):
    agent = DemoAgent(FakeRisk(), None, work_dir=tmp_path)
    assert isinstance(build_ui(agent, CASES, "note", ask_fn=None, ask_off_reason="no key"), gr.Blocks)
    assert isinstance(build_ui(agent, CASES, "note", ask_fn=lambda q: {}), gr.Blocks)


# ---------------------------------------------------------------- scripts/build_space.py
def _build_space():
    spec = importlib.util.spec_from_file_location("build_space", PROJECT_ROOT / "scripts" / "build_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_space_requirements_are_pinned_from_the_lock(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("Gradio==6.0.0\npython_dotenv==1.0.0\nother==1\n")
    bs = _build_space()
    assert bs.pinned_requirements(lock, ["gradio", "python-dotenv"]) == "Gradio==6.0.0\npython_dotenv==1.0.0\n"
    with pytest.raises(SystemExit, match="missing-pkg"):
        bs.pinned_requirements(lock, ["missing-pkg"])


def test_space_build_refuses_files_with_keys(tmp_path):
    bs = _build_space()
    (tmp_path / "ok.txt").write_text("api_key_env: NVIDIA_API_KEY\n")
    bs.check_no_secrets(tmp_path)
    (tmp_path / ".env").write_text("NVIDIA_API_KEY=nvapi-" + "x" * 40 + "\n")
    with pytest.raises(SystemExit, match=".env"):
        bs.check_no_secrets(tmp_path)


def _module_level_imports(path: Path) -> set[str]:
    """Top-level names imported when the module is LOADED (not inside functions: those run only if called)."""
    names, stack = set(), list(ast.parse(path.read_text()).body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
        stack.extend(ast.iter_child_nodes(node))
    return names


def test_space_packages_cover_what_the_demo_imports():
    """Every third-party package our modules import when the demo (and the live RAG) starts must be one of
    SPACE_PACKAGES. Lazy imports inside functions the demo never calls (xgboost, mlflow in train.py) do not
    count. Dependencies of those packages are pip's job (and the lock file's)."""
    bs = _build_space()

    # Step 1: which of OUR modules are loaded at start-up
    code = ("import sys, pancreasignal.demo, pancreasignal.rag.answer, pancreasignal.rag.index\n"
            "print(' '.join(m for m in sys.modules if m.split('.')[0] == 'pancreasignal'))")
    ours = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                          cwd=PROJECT_ROOT).stdout.split()

    # Step 2: their module-level third-party imports vs the Space's package list
    imported = set().union(*(_module_level_imports(Path(importlib.util.find_spec(m).origin)) for m in ours))
    third_party = imported - set(sys.stdlib_module_names) - {"pancreasignal", "__future__"}
    import_name = {"scikit-learn": "sklearn", "pyyaml": "yaml", "python-dotenv": "dotenv"}
    assert third_party - {import_name.get(p, p) for p in bs.SPACE_PACKAGES} == set()
