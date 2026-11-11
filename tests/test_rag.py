"""Phase 8 — guideline RAG. No network: fake LLM client, tiny JATS XML, LanceDB in a temp folder."""
import json
import types

import numpy as np
import pytest

from pancreasignal.rag import answer as answer_mod
from pancreasignal.rag.answer import NON_ANSWER, answer_question, parse_json, validate_answer
from pancreasignal.rag.chunking import chunk_article
from pancreasignal.rag.evaluate import first_gold_rank, parse_judge, retrieval_summary, summarize
from pancreasignal.rag.llm import LLMClient

META = {"pmcid": "PMC1", "title": "Test article", "journal": "J Test", "year": 2020, "license": "cc by", "topic": "ca19_9"}

JATS = """<article><front><article-meta><abstract><title>Abstract</title>
<p>CA19-9 is the main marker for pancreatic cancer and is used in diagnosis and follow up of many patients.</p>
</abstract></article-meta></front>
<body>
<sec><title>Results</title>
  <sec><title>Jaundice</title>
    <p>CA19-9 rises in benign obstructive jaundice<sup><xref ref-type="bibr" rid="R1">1</xref>,<xref ref-type="bibr" rid="R2">2</xref></sup> and
    falls after drainage, as La Greca et al [3] reported in a cohort of patients [57.7%] with long follow up.</p>
    <table-wrap><caption><p>CA19-9 levels by bilirubin group</p></caption><table><tr><td>99</td></tr></table></table-wrap>
  </sec>
</sec>
<sec><title>Acknowledgements</title><p>We thank everyone who helped with this study and the funding bodies that paid.</p></sec>
</body>
<back><ref-list><ref><p>Reference text that must never be indexed at all in any chunk.</p></ref></ref-list></back>
</article>"""


@pytest.fixture
def xml_file(tmp_path):
    path = tmp_path / "PMC1.xml"
    path.write_text(JATS)
    return path


# ---------------------------------------------------------------- chunking
def test_chunking_sections_and_cleaning(xml_file):
    chunks = chunk_article(xml_file, META, min_words=5)
    sections = [c["section"] for c in chunks]
    assert sections[0] == "Abstract", "no 'Abstract > Abstract'"
    assert "Results > Jaundice" in sections, "nested section path kept"
    text = " ".join(c["text"] for c in chunks)
    assert "jaundice and falls" in text, "tagged citation numbers removed"
    assert "et al reported" in text, "plain-text citation [3] removed"
    assert "[57.7%]" in text, "data in brackets kept"
    assert "Table: CA19-9 levels by bilirubin group" in text and "99" not in text, "caption kept, table cells dropped"
    assert "thank" not in text and "Reference text" not in text, "acknowledgements and references skipped"
    assert chunks[0]["chunk_id"] == "PMC1#000" and chunks[0]["license"] == "cc by"


def test_chunking_respects_max_words(tmp_path):
    sentence = "This sentence about pancreatic cysts has exactly ten words. "
    xml = f"<article><body><sec><title>Long</title><p>{sentence * 60}</p></sec></body></article>"
    path = tmp_path / "PMC2.xml"
    path.write_text(xml)
    chunks = chunk_article(path, {**META, "pmcid": "PMC2"}, target_words=100, max_words=150)
    assert len(chunks) > 1 and all(c["n_words"] <= 150 for c in chunks)


# ---------------------------------------------------------------- answer: JSON, labels, repair, verify
GOOD = '{"insufficient_evidence": false, "sentences": [{"text": "CA19-9 rises in jaundice.", "sources": ["S1"]}]}'
TWO = ('{"sentences": [{"text": "CA19-9 rises in jaundice.", "sources": ["S1"]},'
       ' {"text": "Barrett surveillance is every 3 years.", "sources": ["S2"]}]}')


@pytest.mark.parametrize("data,ok", [
    ({"sentences": [{"text": "x", "sources": ["S2"]}]}, True),
    ({"sentences": [{"text": "x", "sources": ["[S1]", "S2"]}]}, True),     # brackets tolerated
    ({"sentences": [{"text": "x", "sources": ["S3"]}]}, False),            # only S1..S2 exist
    ({"sentences": [{"text": "x", "sources": []}]}, False),                # every sentence needs a source
    ({"insufficient_evidence": True, "sentences": []}, True),
    (None, False),
])
def test_validate_answer(data, ok):
    clean, problem = validate_answer(data, n_sources=2)
    assert (clean is not None) == ok and (problem is None) == ok


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("no json") is None


@pytest.mark.parametrize("text,is_refusal", [
    ("The dose is not specified in the provided sources.", True),
    ("The sources do not mention the dose.", True),
    ("CA19-9 is not elevated in Lewis-negative patients.", False),
    ("CA19-9 alone is not sufficient for screening asymptomatic people.", False),
    ("No survival benefit was shown in the sources' control group comparison.", False),
])
def test_non_answer_detection(text, is_refusal):
    assert bool(NON_ANSWER.search(text)) == is_refusal


class FakeClient:
    """Returns scripted chat replies in order; records the messages it was sent."""
    def __init__(self, replies):
        self.replies, self.sent, self.elapsed_total = list(replies), [], 0.0

    def chat(self, model, messages, **kw):
        self.sent.append(messages)
        return self.replies.pop(0)


@pytest.fixture
def fake_search(monkeypatch):
    chunks = [{"chunk_id": f"PMC1#00{i}", "pmcid": "PMC1", "title": "T", "journal": "J", "year": 2020,
               "section": "S", "text": f"text {i}", "rank": i + 1, "score": 1.0} for i in range(2)]
    monkeypatch.setattr(answer_mod, "search", lambda *a, **k: chunks)
    return chunks


CFG = {"answer": {"model": "m", "max_tokens": 100, "temperature": 0, "extra_body": None}}


def V(*labels_per_claim):
    """A verifier reply: for claim i, the labels that support it."""
    return json.dumps({"claims": [{"claim": i + 1, "supported_by": list(l)} for i, l in enumerate(labels_per_claim)]})


@pytest.mark.parametrize("replies,refused,repaired,n_dropped", [
    ([GOOD, V(["S1"])], False, False, 0),                                                  # good answer
    (['{"insufficient_evidence": true, "sentences": []}'], True, False, 0),               # explicit refusal
    (['{"sentences": [{"text": "Not in the provided sources.", "sources": ["S1"]}]}'], True, False, 0),
    (["FUT2 and FUT3", GOOD, V(["S1"])], False, True, 0),                                 # repaired once
    (["FUT2 and FUT3", "still not json"], True, True, 0),                                 # repair failed
    (['{"sentences": [{"text": "x", "sources": ["S9"]}]}', "nope"], True, True, 0),      # label never retrieved
    ([TWO, V(["S1"], [])], False, False, 1),                                               # invented sentence dropped
    ([GOOD, V([])], True, False, 1),                                                       # nothing supported
    ([GOOD, "garbage"], True, False, 0),                                                  # verifier unusable
    ([GOOD, V(["S1"], ["S2"])], True, False, 0),                                          # wrong number of verdicts
])
def test_answer_pipeline(fake_search, replies, refused, repaired, n_dropped):
    client = FakeClient(replies)
    res = answer_question("q?", table=None, client=client, cfg=CFG)
    assert res["refused"] is refused and res["repaired"] is repaired
    assert len(res["dropped_sentences"]) == n_dropped
    assert client.replies == [], "every scripted reply was used: no unexpected or missing calls"
    if refused:
        assert res["citations"] == [] and "not contain enough evidence" in res["answer"]
    else:
        assert "[PMC1#000]" in res["answer"] and "S1" not in res["answer"], "labels become real chunk ids"


def test_verifier_corrects_a_wrong_label(fake_search):
    # the answer cites S1, but the verifier finds the fact in S2: keep the sentence, cite S2
    res = answer_question("q?", table=None, client=FakeClient([GOOD, V(["S2"])]), cfg=CFG)
    assert not res["refused"] and res["recited_sentences"] == 1
    assert res["answer"].endswith("[PMC1#001]") and [c["chunk_id"] for c in res["citations"]] == ["PMC1#001"]


# ---------------------------------------------------------------- LLM client: cache + retry
class FakeSDK:
    """Mimics openai.OpenAI just enough: chat.completions.create, failing `fail_first` times with 429."""
    def __init__(self, fail_first=0):
        self.calls, self.fail_first = 0, fail_first
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_first:
            err = RuntimeError("rate limited")
            err.status_code = 429
            raise err
        msg = types.SimpleNamespace(content="hello", reasoning_content=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])


def test_client_caches_identical_requests(tmp_path):
    sdk = FakeSDK()
    client = LLMClient("u", "UNUSED", tmp_path, sdk_client=sdk)
    for _ in range(3):
        assert client.chat("m", [{"role": "user", "content": "hi"}]) == "hello"
    assert sdk.calls == 1 and client.calls == {"api": 1, "cache": 2, "retries": 0}
    client.chat("m", [{"role": "user", "content": "different"}])
    assert sdk.calls == 2, "a different request is not served from the cache"


def test_client_retries_rate_limits(tmp_path, monkeypatch):
    monkeypatch.setattr("pancreasignal.rag.llm.time.sleep", lambda s: None)
    client = LLMClient("u", "UNUSED", tmp_path, sdk_client=FakeSDK(fail_first=2))
    assert client.chat("m", [{"role": "user", "content": "hi"}]) == "hello"
    assert client.calls["retries"] == 2


# ---------------------------------------------------------------- LanceDB index
class FakeEmbedClient:
    """Deterministic 8-d unit vectors from the text, so similar texts are NOT required to match."""
    def __init__(self):
        self.calls = {"api": 0, "cache": 0, "retries": 0}

    def embed(self, model, texts, input_type=None, batch_size=32, extra_body=None):
        out = []
        for t in texts:
            v = np.random.default_rng(abs(hash(t.split("\n")[-1])) % 2**32).normal(size=8)
            out.append((v / np.linalg.norm(v)).tolist())
        return out


def test_index_build_search_and_model_check(tmp_path):
    from pancreasignal.rag.index import build_index, open_index, search

    cfg = {"paths": {"index_dir": str(tmp_path / "idx")}, "retrieval": {"mode": "hybrid", "top_k": 2},
           "embedding": {"model": "emb-A", "query_input_type": "query", "passage_input_type": "passage", "batch_size": 8}}
    chunks = [{**META, "chunk_id": f"PMC1#00{i}", "section": "S", "text": t} for i, t in
              enumerate(["CA19-9 rises in obstructive jaundice", "MRI is preferred for cyst surveillance",
                         "Lewis negative people make no CA19-9"])]
    build_index(chunks, FakeEmbedClient(), cfg)
    table = open_index(cfg)
    for mode in ("vector", "hybrid"):
        hits = search(table, FakeEmbedClient(), cfg, "jaundice", mode=mode, k=2)
        assert len(hits) == 2 and hits[0]["rank"] == 1 and "score" in hits[0]
    # BM25 only returns chunks that contain the word: 1 here, even with k=2 (hybrid fills up from vectors)
    assert len(search(table, FakeEmbedClient(), cfg, "jaundice", mode="fts", k=2)) == 1
    assert search(table, FakeEmbedClient(), cfg, "jaundice", mode="fts", k=1)[0]["chunk_id"] == "PMC1#000"
    cfg["embedding"]["model"] = "emb-B"
    with pytest.raises(ValueError, match="rebuild the index"):
        open_index(cfg)


# ---------------------------------------------------------------- evaluation helpers
def test_retrieval_metrics():
    assert first_gold_rank(["A", "B", "C"], ["C", "X"]) == 3
    assert first_gold_rank(["A"], ["X"]) is None
    s = retrieval_summary([1, 3, None, 2], ks=[1, 3])
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.75
    assert s["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 4)


def test_parse_judge():
    good = 'Sure: {"claims": [{"claim": "x", "supported": true}], "key_points": [true, false]}'
    assert parse_judge(good, 2) == {"claims": [{"claim": "x", "supported": True}], "key_points": [True, False]}
    assert parse_judge(good, 3) is None, "wrong number of key points is rejected, not guessed"
    assert parse_judge("no json here", 1) is None


def test_summarize():
    records = [
        {"id": "a", "type": "answerable", "refused": False, "repaired": False, "model_time_s": 1.0, "cites_gold": True,
         "judge_ok": True, "claims": 2, "supported_claims": 2, "key_points_covered": [True, True]},
        {"id": "b", "type": "answerable", "refused": True, "repaired": True, "model_time_s": 2.0},
        {"id": "u", "type": "unanswerable", "refused": True, "repaired": False, "model_time_s": 3.0},
    ]
    s = summarize(records)
    assert s["answer_rate"] == 0.5 and s["false_refusals"] == ["b"]
    assert s["fully_correct"] == 0.5 and s["claims_supported"] == 1.0
    assert s["refusal_rate_unanswerable"] == 1.0 and s["repair_rate"] == pytest.approx(1 / 3)


def test_cache_key_ignores_int_vs_float(tmp_path):
    sdk = FakeSDK()
    client = LLMClient("u", "UNUSED", tmp_path, sdk_client=sdk)
    client.chat("m", [{"role": "user", "content": "hi"}], temperature=0)
    client.chat("m", [{"role": "user", "content": "hi"}], temperature=0.0)
    assert sdk.calls == 1, "temperature 0 and 0.0 are the same request"


def test_numbers_must_be_in_a_cited_source():
    from pancreasignal.rag.answer import fix_numbers, numbers_in

    chunks = [{"text": "Discussion without the figures."}, {"text": "Ratio of sensitivity 1.54 (1.31–1.81); DOR 3.50."}]
    assert numbers_in("sensitivity 1.54 (1.31-1.81)") == ["1.54", "1.31", "1.81"]
    assert numbers_in("CA19-9 of 29,512 U/mL; FUT2 and TP53") == ["29512"], "digits in names are not data"
    # cited chunk 0 lacks the numbers -> chunk 1, which has them, is added
    assert fix_numbers({"text": "Sensitivity ratio 1.54, DOR 3.50.", "sources": [0]}, chunks) == [0, 1]
    # a number that no retrieved chunk contains -> invented -> None (the sentence is dropped)
    assert fix_numbers({"text": "Surveillance every 3 years.", "sources": [0]}, chunks) is None


def test_near_duplicate_sentences_are_dropped(fake_search):
    # the real pair from the q01 answer (evaluation run 4)
    a = "Koprowski first described CA19-9 in 1979 using a mouse monoclonal antibody in a colorectal carcinoma cell line."
    b = "Koprowski first described it in 1979 using a mouse monoclonal antibody in a colorectal carcinoma cell line."
    dup = json.dumps({"sentences": [{"text": a, "sources": ["S1"]}, {"text": b, "sources": ["S1"]}]})
    fake_search[0]["text"] = b
    res = answer_question("q?", table=None, client=FakeClient([dup, V(["S1"], ["S1"])]), cfg=CFG)
    assert res["answer"].count("1979") == 1
