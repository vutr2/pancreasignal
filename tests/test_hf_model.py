"""scripts/build_hf_model.py: the Hugging Face model repo. The model card's code example must run and
give exactly what the API serves, and every number in the card must come from the model's metadata."""
import contextlib
import io
import json
import re

import pytest

from pancreasignal.config import PROJECT_ROOT

MODEL = PROJECT_ROOT / "models" / "urine_v1.joblib"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="models/urine_v1.joblib not exported")


@pytest.fixture
def hf_model(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    import build_hf_model
    return build_hf_model


@needs_model
def test_model_card_example_runs_and_matches_the_served_model(hf_model, tmp_path, monkeypatch):
    """Run the card's Python block as a user would (hf_hub_download -> the local JSON); its pipeline
    must predict what api/bundle.predict_row serves, and its printed output must match the comment."""
    hub = pytest.importorskip("huggingface_hub")
    from pancreasignal.api.bundle import load_bundle, predict_row
    from pancreasignal.api.light import export_light

    # Step 1: the built card + its JSON, with the download pointed at the local file
    bundle = load_bundle(MODEL)
    light_path = tmp_path / "model_light.json"
    light_path.write_text(json.dumps(export_light(bundle)))
    card = hf_model.model_card(export_light(bundle), repo_id="someone/some-model")
    calls = []
    monkeypatch.setattr(hub, "hf_hub_download", lambda repo, name: calls.append((repo, name)) or str(light_path))

    # Step 2: run the example, capturing what it prints
    code, namespace, out = hf_model.usage_code(card), {}, io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "model_card_example", "exec"), namespace)  # noqa: S102 - our own card's example
    assert calls == [("someone/some-model", "model_light.json")]

    # Step 3: same probability as the served model, and the comment in the card tells the truth
    ex = hf_model.EXAMPLE
    served = predict_row(bundle, ex["inputs"], ex["age"], ex["sex"])["probability"]
    assert namespace["model"].predict_proba(namespace["X"])[0, 1] == pytest.approx(served, rel=1e-12)
    printed = out.getvalue().strip()
    assert printed == f"{served:.3f}"
    assert re.search(rf"print\(.*\)\s+# {re.escape(printed)}\n", code)


@needs_model
def test_model_card_numbers_come_from_the_metadata(hf_model):
    from pancreasignal.api.bundle import load_bundle
    from pancreasignal.api.light import export_light

    light = export_light(load_bundle(MODEL))
    card = hf_model.model_card(light)
    perf = light["meta"]["performance_reference"]
    assert "${" not in card                                          # every placeholder filled
    assert f"**{perf['auroc']:.3f}**" in card
    assert f"{perf['auroc_ci95'][0]:.3f} – {perf['auroc_ci95'][1]:.3f}" in card
    assert f"scikit-learn {light['meta']['sklearn_version']}" in card
    assert card.startswith("---\nlibrary_name: sklearn\n")              # Hub metadata block first


@needs_model
def test_build_writes_only_the_allow_listed_files(hf_model, tmp_path):
    out = tmp_path / "hf_model"
    hf_model.build(out=out, repo_id="someone/some-model")
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "model_light.json", "urine_v1.joblib", "urine_v1.meta.json"]
    assert (out / "urine_v1.joblib").read_bytes() == MODEL.read_bytes()
    assert "someone/some-model" in (out / "README.md").read_text()


def test_card_template_has_no_unknown_placeholders(hf_model):
    """Every ${name} in the template is one model_card() fills (a typo would only fail at build time)."""
    names = set(re.findall(r"\$\{(\w+)\}", hf_model.TEMPLATE.read_text()))
    filled = {"repo_id", "auroc", "auroc_lo", "auroc_hi", "sens90", "sens95", "n_train", "n_pdac", "n_rest",
              "pdac_share", "sklearn_version", "python_version", "commit", "commit_short", "example_probability"}
    assert names == filled
