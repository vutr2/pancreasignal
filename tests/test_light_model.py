"""api/light.py: the pure-Python copy of the served model must give the SAME output as the scikit-learn
bundle (it is what the free browser demo runs), and the browser code must not need numpy/sklearn."""
import json
import math
import random
import subprocess
import sys

import pytest
import yaml

from pancreasignal.agents.clinical import CaseInput
from pancreasignal.api.light import (
    LIGHT_FORMAT,
    export_light,
    percentile,
    predict_row_light,
    to_sklearn_bundle,
)
from pancreasignal.config import PROJECT_ROOT

MODEL = PROJECT_ROOT / "models" / "urine_v1.joblib"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="models/urine_v1.joblib not exported")


@pytest.fixture(scope="module")
def bundle():
    from pancreasignal.api.bundle import load_bundle
    return load_bundle(MODEL)


@pytest.fixture(scope="module")
def light(bundle):
    # through JSON, exactly as the browser receives it
    return json.loads(json.dumps(export_light(bundle)))


def _same(a: dict, b: dict) -> None:
    """Two predict_row outputs agree (floats to 1e-9 relative: numpy vs math log1p may differ in the last bit)."""
    assert math.isclose(a["probability"], b["probability"], rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(a["base_log_odds"], b["base_log_odds"], rel_tol=1e-12)
    assert [c["feature"] for c in a["contributions"]] == [c["feature"] for c in b["contributions"]]
    for ca, cb in zip(a["contributions"], b["contributions"], strict=True):
        assert math.isclose(ca["contribution_log_odds"], cb["contribution_log_odds"], rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(ca["model_value"], cb["model_value"], rel_tol=1e-9, abs_tol=1e-12)
        assert ca["training_percentile"] == cb["training_percentile"]
    assert [o["feature"] for o in a["out_of_range"]] == [o["feature"] for o in b["out_of_range"]]
    assert a["warnings"] == b["warnings"]


@needs_model
def test_light_equals_sklearn_on_the_demo_cases(bundle, light):
    from pancreasignal.api.bundle import predict_row

    cases = yaml.safe_load((PROJECT_ROOT / "evals" / "agent_cases.yaml").read_text())["cases"]
    checked = 0
    for c in cases:
        try:
            case = CaseInput.model_validate({k: c[k] for k in ("case_id", "urine", "clinical")})
        except ValueError:
            continue                                  # c07 has no creatinine: never reaches the model
        u = case.urine
        _same(predict_row(bundle, u.in_dataset_units(), u.age, u.sex),
              predict_row_light(light, u.in_dataset_units(), u.age, u.sex))
        checked += 1
    assert checked == 7


@needs_model
def test_light_equals_sklearn_on_random_inputs(bundle, light):
    """300 patients from far below to far above the training ranges (unit mistakes included)."""
    from pancreasignal.api.bundle import predict_row

    rng = random.Random(0)
    for _ in range(300):
        inputs = {m: 10 ** rng.uniform(-5, 5) for m in ("LYVE1", "REG1B", "TFF1")}
        inputs["creatinine"] = 10 ** rng.uniform(-3, 1.5)
        age, sex = rng.uniform(18, 99), rng.choice("MF")
        _same(predict_row(bundle, inputs, age, sex), predict_row_light(light, inputs, age, sex))


@needs_model
def test_rebuilt_sklearn_pipeline_equals_the_pickled_one(bundle, light):
    """to_sklearn_bundle (what the browser's scikit-learn engine runs) = the original .joblib, exactly,
    and without scikit-learn warnings (e.g. about feature names)."""
    import warnings

    from pancreasignal.api.bundle import predict_row

    rebuilt = to_sklearn_bundle(light)
    rng = random.Random(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(300):
            inputs = {m: 10 ** rng.uniform(-5, 5) for m in ("LYVE1", "REG1B", "TFF1")}
            inputs["creatinine"] = 10 ** rng.uniform(-3, 1.5)
            age, sex = rng.uniform(18, 99), rng.choice("MF")
            a, b = predict_row(bundle, inputs, age, sex), predict_row(rebuilt, inputs, age, sex)
            assert a["probability"] == b["probability"]
            _same(a, b)


def test_percentile_matches_numpy():
    import numpy as np

    q = sorted(random.Random(1).uniform(0, 10) for _ in range(101))
    for v in [-1, 0, *q[::7], 5.5, 10, 11]:
        assert percentile(q, v) == float(np.searchsorted(q, v, side="right").clip(1, 101) - 1)


def test_light_rejects_other_files():
    with pytest.raises(ValueError, match="light model"):
        predict_row_light({"format": "something-else"}, {}, 60, "M")
    assert LIGHT_FORMAT.startswith("pancreasignal-light")


def _module_level_imports(path) -> set[str]:
    """Top-level names imported when a module is loaded (imports inside functions do not count)."""
    import ast

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


def test_browser_sklearn_engine_needs_only_pyodide_packages():
    """The scikit-learn engine imports the serving code (api/bundle.py). Our modules it loads may only
    import what Pyodide ships: not the training code's mlflow/xgboost/shap, not requests or LangGraph.
    (Optional imports inside pandas/joblib, e.g. pyarrow or psutil, are their business: they are skipped
    when missing. The real proof is running it in Pyodide: see deploy/static/README.md.)"""
    import importlib.util
    from pathlib import Path

    code = ("import sys, pancreasignal.browser_demo, pancreasignal.api.bundle\n"
            "print(' '.join(m for m in sys.modules if m.split('.')[0] == 'pancreasignal'))")
    ours = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                          cwd=PROJECT_ROOT).stdout.split()
    imported = set().union(*(_module_level_imports(Path(importlib.util.find_spec(m).origin)) for m in ours))
    third_party = imported - set(sys.stdlib_module_names) - {"pancreasignal", "__future__"}
    assert third_party <= {"pydantic", "numpy", "pandas", "sklearn", "joblib", "yaml"}, third_party
    assert "pancreasignal.train" not in ours and "pancreasignal.models" not in ours


def test_browser_code_needs_only_pydantic():
    """What the browser imports must exist in Pyodide: the standard library + pydantic. No numpy,
    pandas, scikit-learn, LangGraph or Gradio, even indirectly."""
    code = ("import sys, importlib.metadata as md\n"
            "import pancreasignal.browser_demo\n"
            "dists = md.packages_distributions()\n"
            "names = {m.split('.')[0] for m in sys.modules}\n"
            "print(' '.join(sorted({d for m in names for d in dists.get(m, [])})))")
    used = set(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                              cwd=PROJECT_ROOT).stdout.split())
    allowed = {"pydantic", "pydantic_core", "typing_extensions", "annotated-types", "typing-inspection",
               "pancreasignal"}
    assert used - allowed == set()
