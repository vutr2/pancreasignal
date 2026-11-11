"""Turn JATS full-text XML (Europe PMC) into retrieval chunks (Phase 8). Tests: tests/test_chunking.py

JATS = the XML standard for biomedical articles:
    <front>  title, <abstract>
    <body>   nested <sec> (each with a <title>) containing <p>, <list>, <table-wrap>, <fig>
    <back>   references, acknowledgements, funding... -> never indexed

A chunk = consecutive paragraphs of ONE section, ~TARGET_WORDS words, plus its section path
("Recommendations > 4 IPMN > 4.1 What is the risk of ..."). Keeping the section path lets
retrieval match the heading and lets an answer cite "which article, which section".
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

TARGET_WORDS = 300   # close a chunk once it reaches this size
MAX_WORDS = 450      # a single paragraph longer than this is split by sentences
MIN_WORDS = 15       # drop tiny leftovers (e.g. a heading-only section)

# Sections that are not medical content
SKIP_SECTION = re.compile(
    r"acknowledg|funding|conflicts? of interest|competing interest|author.?s?.? contribution|disclosure"
    r"|data availability|abbreviation|supplementary|ethic|consent|declaration|reference", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
# leftovers after removing citation numbers: "levels ( )", "risk [ , ]", " ,", " ."
_EMPTY_BRACKETS = re.compile(r"[\(\[]\s*[,;–\-\s]*[\)\]]")
# citations typed as plain text, not tagged: "[3]", "[1-3]", "[2, 5]". Only digits, commas and
# dashes inside, so data such as "[57.7%]", "[9.5]", "[95% CI ...]" or "[1-13C]" is kept.
_PLAIN_CITATION = re.compile(r"\s?\[\d+(?:\s*[,–\-]\s*\d+)*\]")


def _clean(text: str) -> str:
    """Collapse whitespace and tidy what is left after citation markers were removed."""
    text = _PLAIN_CITATION.sub(" ", text)
    text = _EMPTY_BRACKETS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:)])", r"\1", text)


def _text(elem: ET.Element) -> str:
    """Readable text of an element WITHOUT reference citations (<xref ref-type="bibr">, and a
    <sup> that only wraps such citations). Inline tags like <italic> keep their text."""
    parts: list[str] = []

    def walk(e: ET.Element) -> None:
        # Step 1: skip citation numbers ("1–4", "[12]") entirely, including their wrapping <sup>
        if e.tag == "xref" and e.get("ref-type") == "bibr":
            parts.append(e.tail or "")
            return
        if e.tag == "sup" and e.find(".//xref[@ref-type='bibr']") is not None:
            parts.append(e.tail or "")
            return
        # Step 2: normal element: its text, its children, then the text that follows it
        parts.append(e.text or "")
        for child in e:
            walk(child)
        if e is not elem:
            parts.append(e.tail or "")

    walk(elem)
    return _clean("".join(parts))


def _paragraphs(sec: ET.Element) -> list[str]:
    """Paragraph texts directly in `sec` (not in its sub-sections), in document order."""
    out: list[str] = []
    for child in sec:
        # Step 1: sub-sections and the heading are handled by the caller
        if child.tag in ("sec", "title", "label"):
            continue
        # Step 2: tables and figures: keep the caption (what it shows), skip the numbers
        if child.tag in ("table-wrap", "fig"):
            caption = child.find("caption")
            if caption is not None and _text(caption):
                out.append(f"{'Table' if child.tag == 'table-wrap' else 'Figure'}: {_text(caption)}")
            continue
        # Step 3: everything else (p, list, boxed-text, disp-quote...): every <p> inside, or the text itself
        ps = [child] if child.tag == "p" else list(child.iter("p"))
        out.extend(t for t in (_text(p) for p in ps) if t)
    return out


def _sections(elem: ET.Element, path: list[str]) -> list[tuple[str, list[str]]]:
    """[(section path, paragraphs)] for `elem` and all its sub-sections, depth first."""
    title = elem.find("title")
    heading = _text(title) if title is not None else ""
    # don't repeat the parent's name ("Abstract > Abstract": the abstract often has <title>Abstract</title>)
    repeats_parent = bool(path) and heading.lower() == path[-1].lower()
    here = path + [heading] if heading and not repeats_parent else path
    # Step 1: skip whole non-medical sections (and everything below them)
    if heading and SKIP_SECTION.search(heading):
        return []
    # Step 2: this section's own paragraphs, then its sub-sections in order
    result = [(" > ".join(here), _paragraphs(elem))]
    for sub in elem.findall("sec"):
        result.extend(_sections(sub, here))
    return result


def _split_long(paragraph: str, target_words: int, max_words: int) -> list[str]:
    """Split a paragraph longer than max_words into pieces of whole sentences (~target_words each)."""
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    pieces, current = [], []
    for sentence in _SENTENCE_END.split(paragraph):
        if current and len(" ".join(current + [sentence]).split()) > target_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _pack(paragraphs: list[str], target_words: int, max_words: int) -> list[str]:
    """Join consecutive paragraphs of ONE section into texts of ~target_words, never above max_words."""
    texts: list[str] = []
    current: list[str] = []
    for piece in (piece for p in paragraphs for piece in _split_long(p, target_words, max_words)):
        # Step 1: close the current text first if this piece would push it past max_words
        if current and len(" ".join(current + [piece]).split()) > max_words:
            texts.append(" ".join(current))
            current = []
        # Step 2: add the piece; close the text once it reaches the target size
        current.append(piece)
        if len(" ".join(current).split()) >= target_words:
            texts.append(" ".join(current))
            current = []
    # Step 3: whatever is left at the end of the section
    if current:
        texts.append(" ".join(current))
    return texts


def chunk_article(xml_path: str | Path, meta: dict, target_words: int = TARGET_WORDS,
                  max_words: int = MAX_WORDS, min_words: int = MIN_WORDS) -> list[dict]:
    """All chunks of one article. `meta` = the article's entry in configs/rag_corpus.yaml
    (pmcid, title, journal, year, license, topic). Sizes come from configs/rag.yaml `chunking`."""
    root = ET.parse(xml_path).getroot()

    # Step 1: abstract first (dense with facts), then the body; <back> is never read
    sections: list[tuple[str, list[str]]] = []
    for abstract in root.findall("front/article-meta/abstract"):
        sections.extend((path or "Abstract", ps) for path, ps in _sections(abstract, ["Abstract"]))
    body = root.find("body")
    if body is not None:
        sections.extend(_sections(body, []))

    # Step 2: within each section, pack paragraphs into ~target_words chunks
    chunks = [{"section": section or "Body", "text": text}
              for section, paragraphs in sections
              for text in _pack(paragraphs, target_words, max_words)
              if len(text.split()) >= min_words]

    # Step 3: ids and metadata (what the answer needs to cite the source)
    return [{
        "chunk_id": f"{meta['pmcid']}#{i:03d}",
        "pmcid": meta["pmcid"], "title": meta["title"], "journal": meta.get("journal"),
        "year": meta.get("year"), "license": meta["license"], "topic": meta.get("topic"),
        "section": c["section"], "text": c["text"], "n_words": len(c["text"].split()),
    } for i, c in enumerate(chunks)]
