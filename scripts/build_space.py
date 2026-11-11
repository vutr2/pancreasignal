"""Assemble the Hugging Face Space (Docker SDK) for the web demo in dist/space/.

    python scripts/build_space.py                   # evidence snapshot + every file the Space needs
    python scripts/build_space.py --skip-evidence   # reuse reports/demo/evidence_snapshot.json
    hf upload <hf-username>/pancreasignal dist/space . --repo-type space   # publish (README, "Web demo")
    python scripts/build_space.py --set-secret <hf-username>/pancreasignal  # .env key -> Space secret

The Space gets ONLY what is listed in build() (no .env, no raw data, no reports, no tests, no LLM cache):
    Dockerfile, README.md (Space card)          deploy/space/ (+ the corpus attribution table)
    requirements.txt, requirements-lock.txt     versions pinned from this venv's lock file
    src/pancreasignal/                          the package
    models/urine_v1.joblib + .meta.json         coefficients, ranges, percentiles (no patient rows)
    evals/agent_cases.yaml                      the synthetic cases
    reports/demo/evidence_snapshot.json         cited answers to critic.RAG_QUESTIONS
    configs/rag.yaml + data/processed/lancedb/  for the live "Ask the guidelines" tab
    docs/results_summary.md + docs/figures/     the README results block and its figures
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

from pancreasignal.agents.critic import RAG_QUESTIONS
from pancreasignal.config import PROJECT_ROOT, load_config

OUT = PROJECT_ROOT / "dist" / "space"
SNAPSHOT = PROJECT_ROOT / "reports" / "demo" / "evidence_snapshot.json"
TEMPLATES = PROJECT_ROOT / "deploy" / "space"
# what the demo imports (+ the live RAG); pip resolves the rest, constrained by the lock file
SPACE_PACKAGES = ["gradio", "langgraph", "numpy", "pandas", "scikit-learn", "joblib", "pydantic", "pyyaml",
                  "lancedb", "openai", "python-dotenv"]
# what NVIDIA and Hugging Face keys look like: no file in the Space may contain one
SECRET_PATTERNS = [re.compile(rb"nvapi-[A-Za-z0-9_-]{20,}"), re.compile(rb"hf_[A-Za-z0-9]{30,}")]


def build_snapshot(path: Path) -> dict:
    """Ask the RAG every question the critic can ask; save the cited answers (from the cache if present)."""
    from pancreasignal.agents.graph import make_rag_fn

    # Step 1: the Phase 8 RAG (needs NVIDIA_API_KEY in .env unless every call is already cached)
    cfg = load_config("configs/rag.yaml")
    rag_fn, client = make_rag_fn()

    # Step 2: one answer per question; a refusal here means the evidence would be empty -> stop
    answers = {}
    for q in RAG_QUESTIONS:
        res = rag_fn(q)
        if res["refused"]:
            raise SystemExit(f"The RAG refused a critic question, fix it before publishing: {q!r}")
        answers[q] = {k: res[k] for k in ("question", "answer", "refused", "citations")}
        print(f"  evidence: {q}  ({len(res['citations'])} sources)")

    # Step 3: record what produced it, so a stale snapshot can be recognised
    index_meta = json.loads((PROJECT_ROOT / cfg["paths"]["index_dir"] / "index_meta.json").read_text())
    snapshot = {"meta": {"built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                         "answer_model": cfg["answer"]["model"], "embedding_model": cfg["embedding"]["model"],
                         "index_chunks_sha256": index_meta["chunks_sha256"],
                         "llm_api_calls": client.calls["api"], "llm_cache_hits": client.calls["cache"]},
                "answers": answers}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    return snapshot


def _canon(name: str) -> str:
    """Package names compare like pip does: 'Python_Dotenv' == 'python-dotenv'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_requirements(lock: Path, packages: list[str]) -> str:
    """'name==version' for each package, taken from the lock file (same versions as this venv)."""
    pins = {}
    for line in lock.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if m:
            pins[_canon(m.group(1))] = f"{m.group(1)}=={m.group(2)}"
    missing = [p for p in packages if _canon(p) not in pins]
    if missing:
        raise SystemExit(f"Not in {lock.name}: {missing}. Install them and run: pip freeze --exclude-editable > {lock.name}")
    return "\n".join(pins[_canon(p)] for p in packages) + "\n"


def corpus_table(corpus_yaml: Path) -> str:
    """Attribution table for the Space card: every article in the RAG corpus with its license."""
    rows = ["| Article | Journal, year | License |", "|---|---|---|"]
    for a in yaml.safe_load(corpus_yaml.read_text())["articles"]:
        title = a["title"].replace("|", "/")
        rows.append(f"| [{title}](https://pmc.ncbi.nlm.nih.gov/articles/{a['pmcid']}/) "
                    f"| {a['journal']}, {a['year']} | {a['license'].upper()} |")
    return "\n".join(rows) + "\n"


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    else:
        shutil.copy2(src, dst)


def check_no_secrets(folder: Path) -> None:
    """Refuse to finish if any file looks like it holds an API key (prints only the file name)."""
    for path in folder.rglob("*"):
        if path.is_file() and any(p.search(path.read_bytes()) for p in SECRET_PATTERNS):
            raise SystemExit(f"Possible secret in {path.relative_to(folder)}: not publishing this folder.")


def build(skip_evidence: bool) -> None:
    rag_cfg = load_config("configs/rag.yaml")

    # Step 1: the evidence snapshot (or reuse the last one)
    if skip_evidence:
        if not SNAPSHOT.exists():
            raise SystemExit(f"No {SNAPSHOT.relative_to(PROJECT_ROOT)} yet: run without --skip-evidence once.")
        print(f"reusing {SNAPSHOT.relative_to(PROJECT_ROOT)}")
    else:
        print("building the evidence snapshot ...")
        build_snapshot(SNAPSHOT)

    # Step 2: start from an empty folder (dist/ is a build output, never edited by hand)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Step 3: the files the app needs, at the same relative paths as in the repo
    model = PROJECT_ROOT / "models" / "urine_v1.joblib"
    if not model.exists():
        raise SystemExit("No models/urine_v1.joblib: run python scripts/export_model.py first.")
    for rel in ["src/pancreasignal", "models/urine_v1.joblib", "models/urine_v1.meta.json",
                "evals/agent_cases.yaml", "configs/rag.yaml", rag_cfg["paths"]["index_dir"],
                "requirements-lock.txt"]:
        copy(PROJECT_ROOT / rel, OUT / rel)
    copy(SNAPSHOT, OUT / SNAPSHOT.relative_to(PROJECT_ROOT))

    # Step 4: README results block (+ its figures) for the "Results and limits" tab
    readme = (PROJECT_ROOT / "README.md").read_text()
    block = re.search(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", readme, re.DOTALL).group(0)
    (OUT / "docs").mkdir(exist_ok=True)
    (OUT / "docs" / "results_summary.md").write_text(block + "\n")
    for fig in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", block):
        copy(PROJECT_ROOT / fig, OUT / fig)

    # Step 5: Dockerfile, pinned requirements, Space card with the corpus attribution
    copy(TEMPLATES / "Dockerfile", OUT / "Dockerfile")
    (OUT / "requirements.txt").write_text(pinned_requirements(PROJECT_ROOT / "requirements-lock.txt", SPACE_PACKAGES))
    card = (TEMPLATES / "README.md").read_text().replace("<!-- CORPUS -->",
                                                         corpus_table(PROJECT_ROOT / rag_cfg["corpus"]))
    (OUT / "README.md").write_text(card)

    # Step 6: last safety check, then a summary
    check_no_secrets(OUT)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"\n{OUT.relative_to(PROJECT_ROOT)}: {len(files)} files, {size_mb:.1f} MB, no secrets found")
    print("next: hf upload <hf-username>/pancreasignal dist/space . --repo-type space")


def set_secret(repo_id: str) -> None:
    """Copy NVIDIA_API_KEY from .env into the Space's SECRETS through the Hugging Face API.
    The key goes from this machine straight to the Space settings: never into a file of the Space (its
    files are public on a public Space, and kept in its git history), never printed."""
    import os

    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    # Step 1: the key from .env (or the shell); stop if there is none
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise SystemExit(f"NVIDIA_API_KEY is not set in {PROJECT_ROOT / '.env'}")

    # Step 2: store it as a Space secret (uses the token from `hf auth login`; the Space must exist)
    HfApi().add_space_secret(repo_id, "NVIDIA_API_KEY", key,
                             description="NVIDIA API key for the live guideline RAG tab")
    print(f"secret NVIDIA_API_KEY set on {repo_id}; the Space restarts to use it")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-evidence", action="store_true", help="reuse the existing evidence snapshot")
    ap.add_argument("--set-secret", metavar="HF_USER/SPACE",
                    help="only copy NVIDIA_API_KEY from .env into that Space's secrets, then stop")
    args = ap.parse_args()
    if args.set_secret:
        set_secret(args.set_secret)
    else:
        build(args.skip_evidence)


if __name__ == "__main__":
    main()
