"""Answer a question from the guideline corpus, with citations, or refuse (Phase 8). Tests: tests/test_rag.py

Pipeline, and the failure each step is there for (all found by evaluating on evals/rag_questions.yaml):
  1. retrieve top-k chunks, shown to the model as short labels [S1]..[Sk]
       long ids like PMC7769746#006 were miscopied (cited #025, which was never retrieved)
  2. the model answers with JSON forced by a JSON SCHEMA at the API level (constrained decoding)
       asked only in the prompt, short answers came back as plain text ("cholangiocarcinoma"): 50% repairs
  3. one repair attempt if the reply is still unusable
  4. VERIFY against ALL retrieved sources: for each sentence, WHICH sources state it?
       - no source states it            -> the sentence is dropped (if none is left: refuse)
       - the right source had the wrong label -> the citation is corrected, not the sentence dropped
       a citation that EXISTS is not a citation that SUPPORTS: "Barrett's oesophagus: every 3 years"
       was invented and pinned on an unrelated chunk about FUT genes
  5. refuse if the model says the evidence is insufficient (or writes that as a sentence)

A prompt is a request; the schema and the checks in code are the guarantee.
"""
from __future__ import annotations

import json
import re

from pancreasignal.rag.index import search
from pancreasignal.rag.llm import LLMClient

REFUSAL_MESSAGE = ("The guideline corpus does not contain enough evidence to answer this question. "
                   "Not for clinical use: consult the full guidelines or a specialist.")
# A refusal written as a normal sentence ("X is not specified in the provided sources")
NON_ANSWER = re.compile(
    r"\b(?:not|no)\b[^.]{0,80}\b(?:in|from|within|by) the (?:provided |given )?(?:sources?|context|documents?|texts?)\b(?!')"
    r"|\b(?:the )?(?:provided )?sources? (?:do|does) not (?:contain|mention|provide|specify|address|discuss|include|state)",
    re.IGNORECASE)

SYSTEM_PROMPT = """You answer questions about pancreatic cancer biomarkers, screening and pancreatic cysts \
for clinicians, using ONLY the sources provided. Each source starts with its label, e.g. [S1].

Reply with JSON: {"insufficient_evidence": false, "sentences": [{"text": "...", "sources": ["S1"]}]}

Rules:
- Every sentence lists the label(s) of the source(s) that state it.
- Use only facts stated in the sources. Keep numbers and units exactly as written. At most 5 sentences.
- Write full sentences, even for one-word answers, e.g. {"text": "The genes are FUT2 and FUT3.", "sources": ["S1"]}.
- If a source answers the question, even in different words, answer it.
- Reply {"insufficient_evidence": true, "sentences": []} only if NO source contains the answer, for example
  when the question is about a guideline, organ or disease that no source covers.
- If sources disagree, say so and cite both."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "insufficient_evidence": {"type": "boolean"},
        "sentences": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "sources"]}},
    },
    "required": ["insufficient_evidence", "sentences"],
}

VERIFY_PROMPT = """Below are numbered sources and numbered claims. For EACH claim, list the labels of the sources \
that explicitly state it. Use an empty list if no source states it.
A source does NOT state a claim if: a number differs; the claim names a guideline, study, organ or disease the
source does not mention; or the claim needs knowledge that is not in the source.

Sources:
{sources}

Claims:
{claims}

Reply with JSON, one entry per claim, in order: {{"claims": [{{"claim": 1, "supported_by": ["S2"]}}]}}"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "integer"}, "supported_by": {"type": "array", "items": {"type": "string"}}},
        "required": ["claim", "supported_by"]}}},
    "required": ["claims"],
}


def _json_schema(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# names that contain digits are not data: CA19-9, FUT2, TP53, MUC16, BRCA1, CDKN2A, T1N0M0
_NAME_WITH_DIGITS = re.compile(r"\b[A-Za-z][\w\-]*\d[\w\-]*")
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", " ": " ", " ": " "})


def numbers_in(text: str) -> list[str]:
    """Data numbers in `text` ('29512', '1.54', '10'), comparable across texts. Thousands separators are
    removed ('29,512' -> '29512'); digits inside names (CA19-9, FUT2) are ignored."""
    text = _NAME_WITH_DIGITS.sub(" ", text.translate(_DASHES))
    nums = [n.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", n) else n for n in _NUMBER.findall(text)]
    return list(dict.fromkeys(nums))


def fix_numbers(sentence: dict, chunks: list[dict]) -> list[int] | None:
    """Every number in the sentence must appear in a source it cites. Numbers are where a wrong citation
    hurts most (Phase 8: a sentence with '1.54; 3.50; 0.97' was pinned on a chunk without them).
    Numbers are compared WHOLE ('3' does not match '3.50').
    Returns the (possibly extended) source indexes, or None if a number appears in NO retrieved chunk."""
    sources = list(sentence["sources"])
    nums_of = [set(numbers_in(c["text"])) for c in chunks]
    # Step 1: which numbers are not in the cited sources?
    missing = [n for n in numbers_in(sentence["text"]) if not any(n in nums_of[i] for i in sources)]
    # Step 2: cite the other retrieved chunks that contain them; a number found nowhere is invented
    for n in missing:
        holders = [i for i, nums in enumerate(nums_of) if n in nums]
        if not holders:
            return None
        if not set(holders) & set(sources):
            sources.append(holders[0])
    return sources


def _near_duplicate(a: str, b: str) -> bool:
    """Same sentence written twice ('described CA19-9 in 1979' / 'described it in 1979')."""
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.8


def format_sources(chunks: list[dict]) -> str:
    """The retrieved chunks as the model sees them: short label, citation line, section, text."""
    return "\n\n".join(
        f"[S{i + 1}] {c['title']} ({c['journal']}, {c['year']}). Section: {c['section']}\n{c['text']}"
        for i, c in enumerate(chunks))


def parse_json(text: str) -> dict | None:
    """The first JSON object in `text`, or None. Models sometimes wrap JSON in prose or ```json fences."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def label_indexes(labels, n_sources: int) -> list[int]:
    """['S2', '[S1]', 'S9'] -> [1, 0] (0-based, unique, only labels that exist)."""
    idx = [int(m.group(1)) - 1 for lab in labels or [] if (m := re.fullmatch(r"\s*\[?S(\d+)\]?\s*", str(lab)))]
    return [i for i in dict.fromkeys(idx) if 0 <= i < n_sources]


def validate_answer(data: dict | None, n_sources: int) -> tuple[dict | None, str | None]:
    """(clean answer, None) or (None, what is wrong). Clean = insufficient flag + sentences whose
    'sources' are converted from labels ('S2') to 0-based indexes into the retrieved chunks."""
    if not isinstance(data, dict):
        return None, "the reply was not valid JSON"
    if data.get("insufficient_evidence") is True:
        return {"insufficient": True, "sentences": []}, None
    sentences = data.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        return None, "'sentences' is missing or empty"
    clean = []
    for s in sentences:
        text = str(s.get("text", "")).strip() if isinstance(s, dict) else ""
        idx = label_indexes(s.get("sources") if isinstance(s, dict) else [], n_sources)
        if not text or not idx:
            return None, f"a sentence has no valid source label (S1..S{n_sources}): {text[:60]!r}"
        clean.append({"text": text, "sources": idx})
    return {"insufficient": False, "sentences": clean}, None


def verify(sentences: list[dict], chunks: list[dict], client: LLMClient, cfg: dict) -> list[list[int]] | None:
    """For each sentence, the indexes of the retrieved chunks that state it (checked against ALL of
    them, not only the ones the answer cited). None if the reply is unusable."""
    vcfg = cfg["answer"]
    prompt = VERIFY_PROMPT.format(sources=format_sources(chunks),
                                  claims="\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(sentences)))
    raw = client.chat(vcfg["model"], [{"role": "user", "content": prompt}], max_tokens=600, temperature=0.0,
                      extra_body=vcfg.get("extra_body"), response_format=_json_schema("verification", VERIFY_SCHEMA))
    data = parse_json(raw)
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list) or len(claims) != len(sentences):
        return None
    return [label_indexes(c.get("supported_by") if isinstance(c, dict) else [], len(chunks)) for c in claims]


def answer_question(question: str, table, client: LLMClient, cfg: dict, mode: str | None = None) -> dict:
    """Retrieve, answer as schema-constrained JSON, repair once, verify every sentence against all
    retrieved chunks. Returns the answer and everything needed to audit it."""
    acfg = cfg["answer"]
    model_time_before = client.elapsed_total

    # Step 1: retrieve the top-k chunks
    chunks = search(table, client, cfg, question, mode=mode)

    def ask(msgs: list[dict]) -> str:
        return client.chat(acfg["model"], msgs, max_tokens=acfg["max_tokens"], temperature=acfg["temperature"],
                           extra_body=acfg.get("extra_body"), response_format=_json_schema("answer", ANSWER_SCHEMA))

    # Step 2: answer as JSON (shape enforced by the API), sources as short labels
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"}]
    raw = ask(messages)
    answer, problem = validate_answer(parse_json(raw), len(chunks))

    # Step 3: one repair attempt, telling the model exactly what was wrong
    repaired = False
    if answer is None:
        repaired = True
        raw = ask(messages + [{"role": "assistant", "content": raw},
                              {"role": "user", "content": f"Invalid reply: {problem}. Reply again with JSON in the required shape."}])
        answer, problem = validate_answer(parse_json(raw), len(chunks))

    # Step 4: decide: unusable / insufficient / refusal written as a sentence / verify each sentence
    dropped: list[str] = []
    recited = 0
    if answer is None:
        refused, reason = True, f"no valid answer after repair: {problem}"
    elif answer["insufficient"]:
        refused, reason = True, "model said the sources are insufficient"
    elif any(NON_ANSWER.search(s["text"]) for s in answer["sentences"]):
        refused, reason = True, "model wrote a refusal as a sentence ('not in the sources')"
    else:
        support = verify(answer["sentences"], chunks, client, cfg)
        if support is None:
            refused, reason = True, "verification reply was unusable"
        else:
            kept = []
            for s, supporting in zip(answer["sentences"], support, strict=True):
                if not supporting:
                    dropped.append(s["text"])          # no retrieved source states it
                    continue
                if any(_near_duplicate(s["text"], k["text"]) for k in kept):
                    continue                           # the same sentence twice adds nothing
                # every number must be in a cited source (checked in code, not by a model)
                sources = fix_numbers({"text": s["text"], "sources": supporting}, chunks)
                if sources is None:
                    dropped.append(s["text"])          # a number that no retrieved source contains
                    continue
                if set(sources) != set(s["sources"]):
                    recited += 1                       # right fact, wrong label: cite what really supports it
                kept.append({"text": s["text"], "sources": sources})
            answer["sentences"] = kept
            refused = not kept
            reason = "no sentence was supported by any retrieved source" if refused else None

    # Step 5: the result; the shown answer cites real chunk ids (labels are only for the model)
    kept = [] if refused else answer["sentences"]
    cited = list(dict.fromkeys(chunks[i]["chunk_id"] for s in kept for i in s["sources"]))
    by_id = {c["chunk_id"]: c for c in chunks}
    text = " ".join(f"{s['text']} [{', '.join(chunks[i]['chunk_id'] for i in s['sources'])}]" for s in kept)
    return {
        "question": question,
        "answer": REFUSAL_MESSAGE if refused else text,
        "raw_answer": raw,
        "refused": refused,
        "refusal_reason": reason,
        "repaired": repaired,
        "dropped_sentences": dropped,
        "recited_sentences": recited,
        "citations": [{k: by_id[cid][k] for k in ("chunk_id", "pmcid", "title", "journal", "year", "section")}
                      for cid in cited],
        "retrieved": [{k: c[k] for k in ("chunk_id", "pmcid", "rank", "score")} for c in chunks],
        "model_time_s": round(client.elapsed_total - model_time_before, 2),
    }
