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
from pancreasignal.api.light import LIGHT_FORMAT, export_light, percentile, predict_row_light
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


def test_percentile_matches_numpy():
    import numpy as np

    q = sorted(random.Random(1).uniform(0, 10) for _ in range(101))
    for v in [-1, 0, *q[::7], 5.5, 10, 11]:
        assert percentile(q, v) == float(np.searchsorted(q, v, side="right").clip(1, 101) - 1)


def test_light_rejects_other_files():
    with pytest.raises(ValueError, match="light model"):
        predict_row_light({"format": "something-else"}, {}, 60, "M")
    assert LIGHT_FORMAT.startswith("pancreasignal-light")


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
