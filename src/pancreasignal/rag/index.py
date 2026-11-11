"""LanceDB index of the RAG chunks + search (Phase 8). Tests: tests/test_rag.py

LanceDB = a vector database that is just a folder on disk (no server, no Docker), like SQLite.
Three ways to search the same table:
  vector : embed the question, return the chunks whose vectors are closest (meaning)
  fts    : BM25 full-text search (exact words: "CA19-9", "40 mm", gene names)
  hybrid : both, merged by Reciprocal Rank Fusion (RRF): a chunk ranked high by either wins
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import lancedb
from lancedb.index import FTS

from pancreasignal.config import PROJECT_ROOT
from pancreasignal.rag.llm import LLMClient

TABLE = "chunks"
META_FILE = "index_meta.json"
RETURN_COLUMNS = ["chunk_id", "pmcid", "title", "journal", "year", "license", "topic", "section", "text"]


def search_text(chunk: dict) -> str:
    """What is embedded and full-text indexed: title + section path + text. The headings carry a lot
    of meaning (the European guideline's section titles are the clinical questions themselves)."""
    return f"{chunk['title']}\n{chunk['section']}\n{chunk['text']}"


def build_index(chunks: list[dict], client: LLMClient, cfg: dict) -> Path:
    """Embed every chunk and (re)write the LanceDB table + its full-text index."""
    emb = cfg["embedding"]
    index_dir = PROJECT_ROOT / cfg["paths"]["index_dir"]

    # Step 1: embed as PASSAGES (questions are embedded as 'query' at search time)
    texts = [search_text(c) for c in chunks]
    vectors = client.embed(emb["model"], texts, input_type=emb["passage_input_type"],
                           batch_size=emb["batch_size"], extra_body=emb.get("extra_body"))

    # Step 2: one row per chunk = metadata + text + vector
    rows = [{**{k: c[k] for k in RETURN_COLUMNS}, "year": int(c["year"] or 0), "search_text": t, "vector": v}
            for c, t, v in zip(chunks, texts, vectors, strict=True)]

    # Step 3: overwrite the table, then build the BM25 index on the same text that was embedded
    db = lancedb.connect(index_dir)
    table = db.create_table(TABLE, data=rows, mode="overwrite")
    # (create_fts_index is deprecated since lancedb 0.25; this is the current API)
    table.create_index("search_text", config=FTS(), replace=True)

    # Step 4: remember HOW the index was built. Searching with a different embedding model than the
    #   one used here gives garbage results without any error, so open_index() checks this.
    meta = {"embedding_model": emb["model"], "dim": len(vectors[0]), "n_chunks": len(rows),
            "chunks_sha256": hashlib.sha256("\n".join(c["chunk_id"] + c["text"] for c in chunks).encode()).hexdigest(),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    return index_dir


def open_index(cfg: dict):
    """The LanceDB table, after checking it was built with the embedding model in the config."""
    index_dir = PROJECT_ROOT / cfg["paths"]["index_dir"]
    meta_path = index_dir / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"No index in {index_dir}. Run: python scripts/build_rag_index.py")
    meta = json.loads(meta_path.read_text())
    if meta["embedding_model"] != cfg["embedding"]["model"]:
        raise ValueError(f"Index was built with {meta['embedding_model']} but the config says "
                         f"{cfg['embedding']['model']}: rebuild the index (scripts/build_rag_index.py)")
    return lancedb.connect(index_dir).open_table(TABLE)


def search(table, client: LLMClient, cfg: dict, question: str, mode: str | None = None,
           k: int | None = None) -> list[dict]:
    """Top-k chunks for `question`, best first. Each result has RETURN_COLUMNS + 'rank' + 'score'."""
    mode = mode or cfg["retrieval"]["mode"]
    k = k or cfg["retrieval"]["top_k"]
    emb = cfg["embedding"]

    # Step 1: the question vector (only vector and hybrid need it)
    query_vec = None
    if mode in ("vector", "hybrid"):
        query_vec = client.embed(emb["model"], [question], input_type=emb["query_input_type"],
                                 extra_body=emb.get("extra_body"))[0]

    # Step 2: run the chosen search
    if mode == "vector":
        query, score_col = table.search(query_vec), "_distance"
    elif mode == "fts":
        query, score_col = table.search(question, query_type="fts", fts_columns="search_text"), "_score"
    elif mode == "hybrid":
        query, score_col = table.search(query_type="hybrid").vector(query_vec).text(question), "_relevance_score"
    else:
        raise ValueError(f"Unknown retrieval mode {mode!r}: vector | fts | hybrid")
    hits = query.limit(k).select(RETURN_COLUMNS).to_list()

    # Step 3: tidy output: rank 1..k and one 'score' column whatever the mode
    return [{**{c: h[c] for c in RETURN_COLUMNS}, "rank": i + 1, "score": float(h[score_col])}
            for i, h in enumerate(hits)]
