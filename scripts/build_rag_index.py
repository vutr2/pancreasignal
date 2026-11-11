"""Build the RAG index (Phase 8): XML -> chunks (jsonl) -> embeddings -> LanceDB.

    python scripts/download_corpus.py      # once: the 20 open-license articles
    python scripts/build_rag_index.py

Re-running is cheap: embeddings come from the cache unless a chunk's text changed.
"""
from __future__ import annotations

import json
from collections import Counter
from statistics import mean

import yaml

from pancreasignal.config import PROJECT_ROOT, load_config
from pancreasignal.rag.chunking import chunk_article
from pancreasignal.rag.index import build_index
from pancreasignal.rag.llm import client_from_config


def main() -> None:
    # Step 1: configs
    cfg = load_config("configs/rag.yaml")
    corpus = yaml.safe_load((PROJECT_ROOT / cfg["corpus"]).read_text())
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]

    # Step 2: chunk every downloaded article (stop if one is missing: the corpus must be complete)
    chunks = []
    for art in corpus["articles"]:
        xml = raw_dir / f"{art['pmcid']}.xml"
        if not xml.exists():
            raise FileNotFoundError(f"{xml} missing. Run: python scripts/download_corpus.py")
        chunks.extend(chunk_article(xml, art, **cfg["chunking"]))

    # Step 3: save the chunks (one JSON per line) so they can be read and checked by hand
    out = PROJECT_ROOT / cfg["paths"]["chunks"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks))

    # Step 4: embed + write the LanceDB table and its BM25 index
    client = client_from_config(cfg)
    index_dir = build_index(chunks, client, cfg)

    per_topic = Counter(c["topic"] for c in chunks)
    print(f"{len(chunks)} chunks from {len(corpus['articles'])} articles | mean {mean(c['n_words'] for c in chunks):.0f} words"
          f" | per topic {dict(per_topic)}")
    print(f"chunks -> {out.relative_to(PROJECT_ROOT)} | index -> {index_dir.relative_to(PROJECT_ROOT)}")
    print(f"API calls {client.calls['api']}, from cache {client.calls['cache']}, retries {client.calls['retries']}")


if __name__ == "__main__":
    main()
