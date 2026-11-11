"""Assemble the Hugging Face MODEL repo for the served urine model in dist/hf_model/.

    python scripts/build_hf_model.py
    hf upload <hf-username>/pancreasignal-urine-logreg dist/hf_model . --repo-type model

The repo gets ONLY these files (allow-list; no data, no .env):
    README.md            model card: deploy/model/README.md, numbers filled in from the model's metadata
    model_light.json     the fitted numbers as JSON (api/light.py export_light): loads in any scikit-learn
    urine_v1.joblib      the original pickled bundle (needs the exact scikit-learn that wrote it)
    urine_v1.meta.json   its SHA-256 + training metadata

The model card's code example is run by tests/test_hf_model.py, so it cannot go stale.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from string import Template

from build_space import check_no_secrets  # scripts/ is on sys.path

from pancreasignal.api.bundle import load_bundle
from pancreasignal.api.light import export_light, predict_row_light
from pancreasignal.config import PROJECT_ROOT

MODEL = PROJECT_ROOT / "models" / "urine_v1.joblib"
TEMPLATE = PROJECT_ROOT / "deploy" / "model" / "README.md"
OUT = PROJECT_ROOT / "dist" / "hf_model"
REPO_ID = "trungvu0307/pancreasignal-urine-logreg"
# the card's example patient (c02 in evals/agent_cases.yaml, synthetic), in the training data's units
EXAMPLE = {"inputs": {"LYVE1": 5.6, "REG1B": 123.0, "TFF1": 723.0, "creatinine": 0.724}, "age": 67, "sex": "M"}


def model_card(light: dict, repo_id: str = REPO_ID) -> str:
    """deploy/model/README.md with every ${...} filled in from the model's own metadata."""
    # Step 1: the numbers, from the metadata saved with the model (never typed by hand)
    meta = light["meta"]
    perf = meta["performance_reference"]
    lo, hi = perf["auroc_ci95"]
    example = predict_row_light(light, EXAMPLE["inputs"], EXAMPLE["age"], EXAMPLE["sex"])["probability"]
    values = {
        "repo_id": repo_id,
        "auroc": f"{perf['auroc']:.3f}", "auroc_lo": f"{lo:.3f}", "auroc_hi": f"{hi:.3f}",
        "sens90": f"{perf['sens_at_spec90']:.1%}", "sens95": f"{perf['sens_at_spec95']:.1%}",
        "n_train": meta["n_train"], "n_pdac": meta["n_pdac"], "n_rest": meta["n_train"] - meta["n_pdac"],
        "pdac_share": f"{meta['n_pdac'] / meta['n_train']:.0%}",
        "sklearn_version": meta["sklearn_version"], "python_version": meta["python_version"],
        "commit": meta["git"]["commit"], "commit_short": meta["git"]["commit"][:7],
        "example_probability": f"{example:.3f}",
    }

    # Step 2: fill the template; substitute() raises KeyError if a ${name} has no value
    return Template(TEMPLATE.read_text()).substitute(values)


def usage_code(card: str) -> str:
    """The first ```python block of the card (the example the tests run)."""
    return re.search(r"```python\n(.*?)```", card, re.DOTALL).group(1)


def build(out: Path = OUT, repo_id: str = REPO_ID) -> None:
    # Step 1: the model must be exactly what the API serves, from committed code
    bundle = load_bundle(MODEL)
    if bundle["meta"]["git"]["uncommitted_changes"]:
        raise SystemExit("The model was trained from uncommitted code: retrain it from a commit first.")
    light = export_light(bundle)

    # Step 2: start from an empty folder, then the allow-listed files
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "README.md").write_text(model_card(light, repo_id))
    (out / "model_light.json").write_text(json.dumps(light, indent=1) + "\n")
    shutil.copy2(MODEL, out / MODEL.name)
    shutil.copy2(MODEL.with_suffix(".meta.json"), out / MODEL.with_suffix(".meta.json").name)

    # Step 3: no keys anywhere, then a summary
    check_no_secrets(out)
    files = sorted(p.name for p in out.iterdir())
    shown = out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out
    print(f"{shown}: {', '.join(files)} (no secrets found)")
    print(f"publish: hf upload {repo_id} {shown} . --repo-type model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=REPO_ID, help="the Hugging Face model repo (user/name)")
    build(repo_id=parser.parse_args().repo_id)


if __name__ == "__main__":
    main()
