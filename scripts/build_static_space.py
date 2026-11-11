"""Assemble the FREE Hugging Face Space (static: runs in the visitor's browser) in dist/static_space/.

    python scripts/build_static_space.py
    hf upload <hf-username>/pancreasignal dist/static_space . --repo-type space

Why static: Gradio and Docker Spaces need HF PRO (creating one returned 402 on 2026-11-11); static
Spaces are free. A static Space has no server, so Pyodide runs the project's Python in the browser
(src/pancreasignal/browser_demo.py). No server also means no secret: there is no API key and no
live RAG question box here; the agent's evidence comes from the snapshot.

The Space gets ONLY these files (allow-list; no .env, no raw data, no pickle):
    index.html, README.md (Space card)    deploy/static/ (+ the corpus attribution table)
    pancreasignal.zip                     exactly the modules browser_demo imports (found by importing it)
    model_light.json                      the model as plain numbers (api/light.py export_light)
    evidence_snapshot.json                cited answers to critic.RAG_QUESTIONS (from scripts/build_space.py)
    cases.json                            the synthetic cases (evals/agent_cases.yaml)
    results.md + docs/figures/            the README results block; relative links -> GitHub
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml
from build_space import (  # scripts/ is on sys.path
    SNAPSHOT,
    build_snapshot,
    check_no_secrets,
    corpus_table,
)

from pancreasignal.api.bundle import load_bundle
from pancreasignal.api.light import export_light
from pancreasignal.config import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "dist" / "static_space"
TEMPLATES = PROJECT_ROOT / "deploy" / "static"
REPO_URL = "https://github.com/vutr2/pancreasignal"


def browser_modules() -> list[str]:
    """Source files of every pancreasignal module the browser runs (asked from a fresh Python)."""
    # browser_demo (pure-Python engine) + api/bundle (the scikit-learn engine's serving code)
    code = ("import sys, pancreasignal.browser_demo, pancreasignal.api.bundle\n"
            "print('\\n'.join(sorted(m.__file__ for n, m in sys.modules.items()\n"
            "      if n.split('.')[0] == 'pancreasignal' and getattr(m, '__file__', None))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
    return out.stdout.split()


def results_for_web(readme_text: str, repo_url: str = REPO_URL) -> tuple[str, list[str]]:
    """(markdown, figure paths): the README results block; images stay relative (copied next to it),
    other relative links point at the GitHub repo."""
    block = re.search(r"<!-- RESULTS:START -->(.*?)<!-- RESULTS:END -->", readme_text, re.DOTALL).group(1)
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", block)
    block = re.sub(r"(?<!!)\[([^\]]*)\]\((?!https?://|#)([^)]+)\)", rf"[\1]({repo_url}/blob/main/\2)", block)
    return "## Results\n\n" + block.strip() + "\n", figures


def build() -> None:
    # Step 1: the evidence snapshot (made by build_space.py; built here if it does not exist yet)
    if not SNAPSHOT.exists():
        print("no evidence snapshot yet: building it ...")
        build_snapshot(SNAPSHOT)

    # Step 2: start from an empty folder
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "docs").mkdir(parents=True)

    # Step 3: the Python the browser runs, zipped with paths relative to src/ (pancreasignal/...)
    src = PROJECT_ROOT / "src"
    modules = browser_modules()
    with zipfile.ZipFile(OUT / "pancreasignal.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in modules:
            z.write(path, arcname=str(Path(path).relative_to(src)))
    print(f"pancreasignal.zip: {len(modules)} modules")

    # Step 4: data files: the model as numbers, the evidence, the cases
    light = export_light(load_bundle(PROJECT_ROOT / "models" / "urine_v1.joblib"))
    (OUT / "model_light.json").write_text(json.dumps(light, indent=1) + "\n")
    shutil.copy2(SNAPSHOT, OUT / "evidence_snapshot.json")
    cases = yaml.safe_load((PROJECT_ROOT / "evals" / "agent_cases.yaml").read_text())["cases"]
    (OUT / "cases.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n")

    # Step 5: results block + figures, the page and the Space card
    text, figures = results_for_web((PROJECT_ROOT / "README.md").read_text())
    (OUT / "results.md").write_text(text)
    for fig in figures:
        (OUT / fig).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / fig, OUT / fig)
    shutil.copy2(TEMPLATES / "index.html", OUT / "index.html")
    rag_cfg = load_config("configs/rag.yaml")
    (OUT / "README.md").write_text((TEMPLATES / "README.md").read_text().replace(
        "<!-- CORPUS -->", corpus_table(PROJECT_ROOT / rag_cfg["corpus"])))

    # Step 6: no keys anywhere, then a summary
    check_no_secrets(OUT)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"{OUT.relative_to(PROJECT_ROOT)}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.2f} MB, "
          "no secrets found")
    print("test locally: python -m http.server 8000 --directory dist/static_space  -> http://127.0.0.1:8000")
    print("publish:      hf upload <hf-username>/pancreasignal dist/static_space . --repo-type space")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    build()


if __name__ == "__main__":
    main()
