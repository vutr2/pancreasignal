"""Evaluate the guideline RAG on evals/rag_questions.yaml (Phase 8). Tests: tests/test_rag.py

Three layers, each answering a different question:
  1. Retrieval   : is an article that contains the answer among the top-k chunks?  (hit@k, MRR)
                   Measured for vector / BM25 / hybrid. No LLM involved, so this is cheap.
  2. Answer      : did it answer (or correctly refuse), cite a gold article, cover the key points?
  3. Faithfulness: is every claim in the answer supported by the chunks it cites? (LLM judge,
                   a different model family from the answer model)
"""
from __future__ import annotations

import json
import re
import statistics

import yaml

from pancreasignal.config import PROJECT_ROOT
from pancreasignal.rag.answer import answer_question
from pancreasignal.rag.index import search
from pancreasignal.rag.llm import LLMClient

JUDGE_SYSTEM = """You are a strict grader of answers written from medical literature. Reply with JSON only."""

JUDGE_TEMPLATE = """Question: {question}

Reference key points (numbered):
{key_points}

Answer to grade:
{answer}

Sources the answer cites:
{sources}

Tasks:
1. Split the answer into its factual claims. For each claim, decide if the cited sources state it.
   A claim is supported only if the sources say it; numbers must match exactly.
2. For each reference key point, decide if the answer states it (paraphrase is fine, numbers must match).

Reply with this JSON and nothing else:
{{"claims": [{{"claim": "...", "supported": true}}], "key_points": [true, false]}}
The "key_points" list must have exactly {n_points} booleans, in the order given."""


# ---------------------------------------------------------------- small pure helpers (unit-tested)
def load_questions(path: str) -> list[dict]:
    return yaml.safe_load((PROJECT_ROOT / path).read_text())["questions"]


def first_gold_rank(retrieved_pmcids: list[str], gold: list[str]) -> int | None:
    """1-based rank of the first chunk from a gold article, or None if none was retrieved."""
    return next((i + 1 for i, p in enumerate(retrieved_pmcids) if p in gold), None)


def retrieval_summary(ranks: list[int | None], ks: list[int]) -> dict:
    """hit@k = share of questions with a gold article in the top k; MRR = mean of 1/rank (0 if missed)."""
    n = len(ranks)
    out = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return out


def parse_judge(text: str, n_points: int) -> dict | None:
    """The judge's JSON, or None if it is missing or malformed (counted, never guessed)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    claims, points = data.get("claims"), data.get("key_points")
    if not isinstance(claims, list) or not isinstance(points, list) or len(points) != n_points:
        return None
    return {"claims": [c for c in claims if isinstance(c, dict) and "supported" in c],
            "key_points": [bool(p) for p in points]}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


# ---------------------------------------------------------------- the three layers
def evaluate_retrieval(questions: list[dict], table, client: LLMClient, cfg: dict,
                       modes=("vector", "fts", "hybrid"), ks=(1, 3, 6)) -> dict:
    """hit@k and MRR per retrieval mode, on the answerable questions."""
    answerable = [q for q in questions if q["type"] == "answerable"]
    result = {}
    for mode in modes:
        # Step 1: rank of the first gold-article chunk for every question
        ranks = []
        for q in answerable:
            hits = search(table, client, cfg, q["question"], mode=mode, k=max(ks))
            ranks.append(first_gold_rank([h["pmcid"] for h in hits], q["gold_pmcids"]))
        # Step 2: summary + which questions were missed (to read them, not just count them)
        result[mode] = {**retrieval_summary(ranks, list(ks)),
                        "missed": [q["id"] for q, r in zip(answerable, ranks, strict=True) if r is None]}
    return result


def judge_answer(q: dict, result: dict, chunk_text: dict, client: LLMClient, cfg: dict) -> dict | None:
    """Ask the judge model about faithfulness (claims vs cited chunks) and key-point coverage."""
    jcfg = cfg["judge"]
    sources = "\n\n".join(f"[{c['chunk_id']}] {chunk_text[c['chunk_id']]}" for c in result["citations"])
    prompt = JUDGE_TEMPLATE.format(
        question=q["question"], answer=result["answer"], sources=sources, n_points=len(q["key_points"]),
        key_points="\n".join(f"{i + 1}. {p}" for i, p in enumerate(q["key_points"])))
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
    # a reasoning judge can run out of tokens mid-thought: try once more with twice the budget
    for max_tokens in (jcfg["max_tokens"], 2 * jcfg["max_tokens"]):
        try:
            raw = client.chat(jcfg["model"], messages, max_tokens=max_tokens, temperature=jcfg["temperature"],
                              extra_body=jcfg.get("extra_body"), timeout_s=jcfg.get("timeout_s"))
            return parse_judge(raw, len(q["key_points"]))
        except Exception as e:  # noqa: BLE001 - one failed grading must not crash a 50-question run: count it
            print(f"  [judge failed] {q['id']} (max_tokens {max_tokens}): {type(e).__name__}: {str(e)[:100]}")
            if "finish_reason=length" not in str(e):
                return None
    return None


def evaluate_answers(questions: list[dict], table, client: LLMClient, cfg: dict) -> tuple[list[dict], dict]:
    """Answer every question, judge the answered ones, return (per-question records, summary)."""
    chunk_text = {c["chunk_id"]: c["text"] for c in
                  (json.loads(line) for line in (PROJECT_ROOT / cfg["paths"]["chunks"]).read_text().splitlines())}
    records = []
    for q in questions:
        # Step 1: the full pipeline, exactly as a user would get it
        res = answer_question(q["question"], table, client, cfg)
        rec = {"id": q["id"], "type": q["type"], "topic": q["topic"], "question": q["question"],
               "refused": res["refused"], "refusal_reason": res["refusal_reason"], "repaired": res["repaired"],
               "answer": res["answer"], "cited": [c["chunk_id"] for c in res["citations"]],
               "dropped_sentences": res["dropped_sentences"], "recited_sentences": res["recited_sentences"],
               "retrieved": [r["chunk_id"] for r in res["retrieved"]], "model_time_s": res["model_time_s"]}

        # Step 2: answerable + answered -> gold citation, then the judge
        if q["type"] == "answerable" and not res["refused"]:
            rec["cites_gold"] = any(c["pmcid"] in q["gold_pmcids"] for c in res["citations"])
            verdict = judge_answer(q, res, chunk_text, client, cfg)
            rec["judge_ok"] = verdict is not None
            if verdict:
                claims = verdict["claims"]
                rec["claims"] = len(claims)
                rec["supported_claims"] = sum(bool(c["supported"]) for c in claims)
                rec["key_points_covered"] = verdict["key_points"]
                rec["unsupported"] = [c.get("claim") for c in claims if not c["supported"]]
        records.append(rec)

    return records, summarize(records)


def summarize(records: list[dict]) -> dict:
    """Headline numbers from the per-question records."""
    ans = [r for r in records if r["type"] == "answerable"]
    una = [r for r in records if r["type"] == "unanswerable"]
    answered = [r for r in ans if not r["refused"]]
    judged = [r for r in answered if r.get("judge_ok")]
    claims = sum(r["claims"] for r in judged)
    lat = [r["model_time_s"] for r in records]
    return {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        # answerable questions
        "answer_rate": len(answered) / len(ans) if ans else None,          # 1 - false refusals
        "false_refusals": [r["id"] for r in ans if r["refused"]],
        "cites_gold_article": sum(r["cites_gold"] for r in answered) / len(answered) if answered else None,
        "key_point_coverage": statistics.mean(statistics.mean(r["key_points_covered"]) for r in judged) if judged else None,
        "fully_correct": sum(all(r["key_points_covered"]) for r in judged) / len(ans) if ans else None,
        # faithfulness (judged answers only)
        "claims_supported": sum(r["supported_claims"] for r in judged) / claims if claims else None,
        "answers_fully_supported": sum(r["supported_claims"] == r["claims"] for r in judged) / len(judged) if judged else None,
        "judge_failures": [r["id"] for r in answered if not r.get("judge_ok")],
        # unanswerable questions: refusing is the right answer
        "refusal_rate_unanswerable": sum(r["refused"] for r in una) / len(una) if una else None,
        "answered_unanswerable": [r["id"] for r in una if not r["refused"]],
        # behaviour
        "repair_rate": sum(r["repaired"] for r in records) / len(records),
        "answers_with_dropped_sentences": sum(bool(r.get("dropped_sentences")) for r in records) / len(records),
        "answers_with_corrected_citation": sum(bool(r.get("recited_sentences")) for r in records) / len(records),
        # time spent in model calls per question (original API time, also when replayed from the cache)
        "model_time_p50_s": percentile(lat, 0.5), "model_time_p95_s": percentile(lat, 0.95),
    }
