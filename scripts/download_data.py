"""
Download public datasets for PancreaSignal into <project>/data/raw/.

Usage (from project root):
    python scripts/download_data.py              # download all
    python scripts/download_data.py --only urine # just one
    python scripts/download_data.py --force      # re-download

Requirements:
    pip install kaggle requests tqdm   (already in pyproject.toml)
    Kaggle credentials: ~/.kaggle/kaggle.json  (chmod 600)
      or env vars KAGGLE_USERNAME / KAGGLE_KEY
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

# scripts/download_data.py -> project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST = PROJECT_ROOT / "data" / "manifest.json"

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
# Platform table (probe ID -> miRNA name). The series matrix only has miRBase IDs (MIMAT...).
GEO_PLATFORM_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gpl}&targ=self&form=text&view=data"

DATASETS = {
    # Debernardi et al. 2020 (PLOS Med) — urinary LYVE1/REG1B/TFF1 + CA19-9
    "urine": {"type": "kaggle", "slug": "johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer"},
    # Serum miRNA, pancreatic & biliary tract cancers (Toray 3D-Gene array, platform GPL18941)
    "mirna_gse59856": {"type": "geo", "gse": "GSE59856", "platform": "GPL18941"},
    # Serum miRNomes, early-stage cancers (large; ~hundreds of MB)
    "mirna_gse211692": {"type": "geo", "gse": "GSE211692"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(name: str, out_dir: Path, source: str) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[name] = {
        "source": source,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": {p.name: sha256(p) for p in sorted(out_dir.iterdir()) if p.is_file()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def download_kaggle(slug: str, out_dir: Path) -> str:
    from kaggle.api.kaggle_api_extended import KaggleApi  # import late: needs credentials

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)
    return f"https://www.kaggle.com/datasets/{slug}"


def download_url(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)  # only "complete" files get the final name


def download_geo(gse: str, out_dir: Path, platform: str | None = None) -> str:
    # GEO folder convention: GSE59856 -> GSE59nnn
    url = f"{GEO_BASE}/{gse[:-3]}nnn/{gse}/matrix/{gse}_series_matrix.txt.gz"
    download_url(url, out_dir / f"{gse}_series_matrix.txt.gz")
    # Optional: platform table, so results can show miRNA names instead of MIMAT IDs
    if platform:
        download_url(GEO_PLATFORM_URL.format(gpl=platform), out_dir / f"{platform}.txt")
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=DATASETS.keys(), nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in args.only or DATASETS:
        cfg = DATASETS[name]
        out_dir = RAW_DIR / name
        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            print(f"[skip] {name}: already in {out_dir.relative_to(PROJECT_ROOT)}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[get ] {name}")
        try:
            if cfg["type"] == "kaggle":
                source = download_kaggle(cfg["slug"], out_dir)
            else:
                source = download_geo(cfg["gse"], out_dir, cfg.get("platform"))
            update_manifest(name, out_dir, source)
            print(f"[ok  ] {name} -> {out_dir.relative_to(PROJECT_ROOT)}")
        except Exception as e:  # keep going with other datasets
            print(f"[fail] {name}: {e}")

    print(
        "\nMIMIC-IV (credentialed) is NOT downloaded automatically. After PhysioNet approval:\n"
        "  wget -r -N -c -np --user <physionet_user> --ask-password \\\n"
        "       -P data/raw/mimic https://physionet.org/files/mimic-iv/<version>/hosp/\n"
        "Keep it on this Mac only — never upload it or send it to LLM APIs."
    )


if __name__ == "__main__":
    main()
