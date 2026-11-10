"""Load + validate the serum miRNA dataset GSE59856 (Phase 5). Tests: tests/test_geo.py

GSE59856 = 571 serum samples measured on a Toray 3D-Gene array (2555 miRNAs, platform GPL18941).
Groups: healthy control, pancreatic cancer, biliary tract cancer, benign pancreatic/biliary disease,
and 4 other cancers (liver, stomach, esophagus, colon).

What GEO already did to the values (from !Sample_data_processing):
  - background subtraction, log2 scale, quantile normalisation across ALL samples together.
  - undetected miRNAs were replaced with one fixed low value -> about 62% of all values sit on
    that "floor" (DETECTION_FLOOR). Most miRNAs are not detected in most samples.

Note on "PDAC": GEO only says "pancreatic cancer". We call it "pdac" so the task names match the
urine model, but the label is not guaranteed to be ductal adenocarcinoma only.
"""
from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import pandas as pd

from pancreasignal.config import RAW_DIR, resolve

MIRNA_DIR = RAW_DIR / "mirna_gse59856"
DETECTION_FLOOR = 2.612  # value GEO used for "not detected" (the minimum of the whole matrix)

# GEO "disease state" text -> short code used everywhere else
DISEASE_CODES = {
    "healthy control": "control",
    "pancreatic cancer": "pdac",
    "biliary tract cancer": "biliary",
    "benign pancreatic or biliary tract diseases": "benign",
    "liver cancer": "liver",
    "stomach cancer": "stomach",
    "esophagus cancer": "esophagus",
    "colon cancer": "colon",
}

# task name -> (negative codes, positive codes)
MIRNA_TASKS: dict[str, tuple[list[str], list[str]]] = {
    "pdac_vs_control": (["control"], ["pdac"]),
    "pdac_vs_biliary": (["biliary"], ["pdac"]),   # two cancers of the same region: the hard one
    "pdac_vs_benign": (["benign"], ["pdac"]),     # only 21 benign samples -> very wide CIs
}

META_COLUMNS = ["sample_id", "title", "title_prefix", "diagnosis", "age", "sex", "stage", "operability"]


def read_series_matrix(path: str | Path) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Parse a GEO series matrix (.txt.gz).

    Returns (meta, expr):
      meta: {field: one value per sample}, e.g. meta["geo_accession"], meta["disease state"]
      expr: DataFrame, one row per sample (index = GSM id), one column per probe (MIMAT id)
    """
    meta: dict[str, list[str]] = {}
    table_lines: list[str] = []
    in_table = False

    # Step 1: read line by line. "!Sample_..." lines are metadata; the block between
    #   !series_matrix_table_begin and !series_matrix_table_end is the data table.
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                table_lines.append(line)
                continue
            if not line.startswith("!Sample_"):
                continue

            key, *cells = line.rstrip("\n").split("\t")
            cells = [c.strip('"') for c in cells]

            # Step 2: characteristics come as "field: value" in every cell, one line per field.
            #   If one line mixes fields, we cannot trust the columns -> fail loudly.
            if key == "!Sample_characteristics_ch1":
                fields = {c.split(": ", 1)[0] for c in cells}
                if len(fields) != 1:
                    raise ValueError(f"Characteristics line mixes fields: {sorted(fields)}")
                meta[fields.pop()] = [c.split(": ", 1)[1] for c in cells]
            else:
                meta[key.removeprefix("!Sample_")] = cells

    # Step 3: data table: rows = probes, columns = samples -> transpose to one row per sample
    if not table_lines:
        raise ValueError(f"No data table found in {path}")
    expr = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0).T
    return meta, expr


def _clean_stage(stage: str) -> str | None:
    """'cStage IV' -> 'IV', 'pStage IIB' -> 'IIB', 'ypStage IB' -> 'IB', 'uncertain' -> None.
    c = clinical, p = pathological, yp = pathological after neoadjuvant therapy."""
    match = re.fullmatch(r"(?:c|p|yp)?Stage\s+([IV]+[ABC]?)", stage.strip())
    return match.group(1) if match else None


def mirna_columns(df: pd.DataFrame) -> list[str]:
    """The miRNA feature columns (miRBase accession IDs start with MIMAT)."""
    return [c for c in df.columns if str(c).startswith("MIMAT")]


def validate_mirna(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if the data doesn't look like we expect."""
    # Step 1: metadata columns and IDs
    missing = set(META_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df["sample_id"].is_unique:
        raise ValueError("Duplicate sample_id values")

    # Step 2: labels and sex must be known codes
    bad_dx = set(df["diagnosis"]) - set(DISEASE_CODES.values())
    if bad_dx:
        raise ValueError(f"Unknown disease states: {sorted(bad_dx)} -> add them to DISEASE_CODES")
    bad_sex = set(df["sex"]) - {"M", "F"}
    if bad_sex:
        raise ValueError(f"Unexpected sex values: {sorted(bad_sex)}")

    # Step 3: the expression matrix: enough miRNAs, the expected floor, few missing values
    cols = mirna_columns(df)
    if len(cols) < 100:
        raise ValueError(f"Only {len(cols)} miRNA columns found")
    values = df[cols]
    if abs(values.min().min() - DETECTION_FLOOR) > 1e-6:
        raise ValueError(
            f"Minimum value is {values.min().min()}, expected the detection floor {DETECTION_FLOOR}. "
            "The preprocessing may have changed; check DETECTION_FLOOR."
        )
    missing_share = values.isna().to_numpy().mean()
    if missing_share > 0.01:
        raise ValueError(f"{missing_share:.1%} of miRNA values are missing (expected < 1%)")
    return df


def _find_matrix(path: str | Path | None) -> Path:
    if path is not None and resolve(path).exists():
        return resolve(path)
    candidates = sorted(MIRNA_DIR.glob("*_series_matrix.txt.gz"))
    if not candidates:
        raise FileNotFoundError(
            "miRNA dataset not found. Run: python scripts/download_data.py --only mirna_gse59856"
        )
    return candidates[0]


def load_mirna(path: str | Path | None = None) -> pd.DataFrame:
    """One row per sample: metadata columns (META_COLUMNS) + one column per miRNA (MIMAT id)."""
    # Step 1: parse the GEO file
    meta, expr = read_series_matrix(_find_matrix(path))

    # Step 2: tidy metadata. "uncertain" age/stage become missing; sex becomes M/F like the urine data
    df = pd.DataFrame({
        "sample_id": meta["geo_accession"],
        "title": meta["title"],
        # "pancreatic cancer P01" -> "P". Controls come as "S.." or "E.." -> two collection batches?
        "title_prefix": [re.sub(r"\d+$", "", t).split()[-1] for t in meta["title"]],
        "diagnosis": [DISEASE_CODES.get(d, d) for d in meta["disease state"]],
        "age": pd.to_numeric(pd.Series(meta["age"]), errors="coerce"),
        "sex": [{"Male": "M", "Female": "F"}.get(s, s) for s in meta["gender"]],
        "stage": [_clean_stage(s) for s in meta["tumor stage"]],
        "operability": meta["operability"],
    })

    # Step 3: attach the expression values in the SAME sample order (match by GSM id, not position)
    expr = expr.loc[df["sample_id"]].reset_index(drop=True)
    df = pd.concat([df, expr], axis=1)
    return validate_mirna(df)


def select_mirna_task(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    """Keep only rows relevant to `task`; return (rows, binary label y with 1 = pancreatic cancer)."""
    if task not in MIRNA_TASKS:
        raise ValueError(f"Unknown miRNA task {task!r}. Choose from {list(MIRNA_TASKS)}")
    neg, pos = MIRNA_TASKS[task]
    rows = df[df["diagnosis"].isin(neg + pos)].reset_index(drop=True)
    y = rows["diagnosis"].isin(pos).astype(int).rename("y")
    return rows, y


def mirna_names(path: str | Path | None = None) -> dict[str, str]:
    """MIMAT id -> miRNA name (e.g. 'MIMAT0000062' -> 'hsa-let-7a-5p') from the GEO platform table.
    Returns {} if the platform file was not downloaded (names are nice-to-have, not required)."""
    # Step 1: find the platform file (downloaded next to the series matrix)
    file = resolve(path) if path is not None else next(iter(sorted(MIRNA_DIR.glob("GPL*.txt"))), None)
    if file is None or not Path(file).exists():
        return {}

    # Step 2: the table sits between !platform_table_begin and !platform_table_end
    text = Path(file).read_text()
    table = text.split("!platform_table_begin\n", 1)[1].split("!platform_table_end", 1)[0]
    platform = pd.read_csv(io.StringIO(table), sep="\t")
    return dict(zip(platform["ID"], platform["miRNA_ID_LIST"], strict=True))
