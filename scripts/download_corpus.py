"""
Download open-access articles (Europe PMC full-text XML) for the Phase 8 RAG corpus.

Usage (from project root):
    python scripts/download_corpus.py
"""

from __future__ import annotations

import json
import time

import requests
import yaml
from download_data import MANIFEST, PROJECT_ROOT, sha256

CONFIG = PROJECT_ROOT / "configs" / "rag_corpus.yaml"
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "guidelines"
FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

def main() -> None:

    # Step 1 read the articles list and the allowed licenses from YAML
    cfg = yaml.safe_load(CONFIG.read_text())
    allowed = {lic.strip().lower() for lic in cfg['allowed_licenses']}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entrices = {}
    for art in cfg['articles']:
        pmcid = art['pmcid']
        license = art['license'].strip().lower()

        #step 2 Skip articles with disallowed licenses
        if license not in allowed:
            print(f'Skipping {pmcid} with license {license}')
            continue

        url = FULLTEXT_URL.format(pmcid=pmcid)
        dest = OUT_DIR / f"{pmcid}.xml"

        #step 3 Do not allow again if the file already exists
        if dest.exists():
            print(f"[have] {pmcid} already downloaded")
        else:
            #Step 4 Download, stop on HTTP Errors, wait a bit so we don't flood the API
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
            except requests.RequestException as e:
                print(f"[fail] {pmcid}: {e}")
                continue
            dest.write_bytes(r.content)
            print(f"[ok ] {pmcid} -> {dest.relative_to(PROJECT_ROOT)}")
            time.sleep(0.5)

        entrices[pmcid] = {"title": art['title'], "license": license, "url": url, "sha256": sha256(dest)}

        #Step 5 Save every corpus file under manifest['guidelines'], keeping the others datasets
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

        manifest["guidelines"] = entrices
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"\n{len(entrices)} articles recorded in manifest['guidelines']")

if __name__ == "__main__":
    main()

