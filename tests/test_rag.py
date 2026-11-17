"""RAG tests: chunk offsets, retrieval, citation checks, both answerers, optional pgvector."""

import json
import os
from pathlib import Path

import anthropic
import httpx2 as httpx
import pytest

from intake_agent.rag import (
    NOT_FOUND,
    HashingEmbedder,
    InMemoryStore,
    MockAnswerer,
    PgVectorStore,
    RagIndex,
    chunk_source,
    load_corpus,
)
from intake_agent.rag.answer import ClaudeAnswerer, ModelAnswer, finalize, verify_citation

POLICIES = Path(__file__).resolve().parents[1] / "data" / "policies"


@pytest.fixture(scope="module")
def corpus(dataset):
    sources, skipped = load_corpus(dataset, POLICIES)
    return {s.doc_id: s for s in sources}, skipped


@pytest.fixture(scope="module")
def index(corpus):
    idx = RagIndex(HashingEmbedder(), InMemoryStore())
    idx.build(list(corpus[0].values()))
    return idx


def first_form(corpus) -> str:
    return next(d for d, s in corpus[0].items() if s.text.startswith("OUTPATIENT REFERRAL FORM"))


def test_scans_are_skipped_not_indexed(corpus):
    sources, skipped = corpus
    assert skipped, "the fixture set has a scan"
    assert not set(skipped) & set(sources)
    assert sum(d.startswith("policy:") for d in sources) == 5


def test_chunk_offsets_point_at_source_text(corpus):
    for source in corpus[0].values():
        chunks = chunk_source(source, max_chars=200)
        assert chunks
        for c in chunks:
            assert source.text[c.start : c.end] == c.text
            assert c.chunk_id.startswith(source.doc_id + "#")


def test_hashing_embedder_is_deterministic_and_unit_length():
    e = HashingEmbedder(dim=64)
    a, b = e.embed("MRI right knee"), e.embed("MRI right knee")
    assert a == b
    assert sum(v * v for v in a) == pytest.approx(1.0)
    near = sum(x * y for x, y in zip(a, e.embed("knee MRI, right side"), strict=True))
    far = sum(x * y for x, y in zip(a, e.embed("allergy skin testing"), strict=True))
    assert near > far


def test_filters(index, corpus):
    doc = first_form(corpus)
    assert {h.chunk.doc_id for h in index.retrieve("member id", doc_id=doc)} == {doc}
    hits = index.retrieve("expedited review", prefix="policy:")
    assert hits and all(h.chunk.doc_id.startswith("policy:") for h in hits)


def test_mock_answer_cites_exact_span(index, corpus):
    sources = corpus[0]
    doc = first_form(corpus)
    ans = MockAnswerer().answer("What is the member ID?", index.retrieve("member id", doc_id=doc))
    assert ans.grounded
    c = ans.citations[0]
    assert c.doc_id == doc
    assert sources[doc].text[c.start : c.end] == c.quote
    assert ans.tokens_estimated and ans.input_tokens > 0


def test_policy_question_cites_the_right_payer(index, corpus):
    q = "Does Fabrikam Mutual require prior authorization for CPT 45378?"
    ans = MockAnswerer().answer(q, index.retrieve(q, prefix="policy:"))
    assert "no prior authorization needed" in ans.text
    assert ans.citations[0].doc_id == "policy:fabrikam-mutual"
    c = ans.citations[0]
    assert corpus[0][c.doc_id].text[c.start : c.end] == c.quote


def test_mock_says_not_found_instead_of_guessing(index, corpus):
    doc = first_form(corpus)
    q = "What is the patient's blood type?"
    ans = MockAnswerer().answer(q, index.retrieve(q, doc_id=doc))
    assert ans.text == NOT_FOUND
    assert not ans.grounded and ans.citations == []


def test_fabricated_or_unretrieved_citations_are_dropped(index):
    hits = index.retrieve("expedited review Northwind", prefix="policy:")
    real = hits[0].chunk
    quote = real.text.splitlines()[0]
    assert verify_citation(hits, real.chunk_id, quote) is not None
    assert verify_citation(hits, real.chunk_id, "a sentence that is not there") is None
    assert verify_citation(hits, "doc-999#0", quote) is None

    parsed = ModelAnswer.model_validate(
        {"found": True, "answer": "Yes.", "citations": [{"chunk_id": "doc-999#0", "quote": quote}]}
    )
    ans = finalize("q", hits, parsed, "m")
    assert ans.text == NOT_FOUND and ans.dropped_citations == 1


def fake_client(captured: list[dict], payload: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 900, "output_tokens": 60},
            },
        )

    return anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_claude_answerer_request_and_citation_check(index, settings):
    q = "What is the expedited review turnaround at Contoso Care?"
    hits = index.retrieve(q, prefix="policy:")
    contoso = next(h.chunk for h in hits if h.chunk.doc_id == "policy:contoso-care")
    line = next(ln for ln in contoso.text.splitlines() if "Expedited" in ln)
    payload = {
        "found": True,
        "answer": "Contoso Care decides expedited reviews within 48 hours.",
        "citations": [
            {"chunk_id": contoso.chunk_id, "quote": line.lstrip("- ")},
            {"chunk_id": contoso.chunk_id, "quote": "decision within 2 hours"},
        ],
    }
    captured: list[dict] = []
    ans = ClaudeAnswerer(settings, client=fake_client(captured, payload)).answer(q, hits)

    req = captured[0]
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert contoso.chunk_id in req["messages"][0]["content"]
    assert ans.grounded and len(ans.citations) == 1 and ans.dropped_citations == 1
    assert ans.citations[0].doc_id == "policy:contoso-care"
    assert ans.cost_usd == pytest.approx((900 * 0.10 + 60 * 0.50) / 1e6)


def test_claude_answerer_not_found(index, settings):
    hits = index.retrieve("blood type", prefix="policy:")
    payload = {"found": False, "answer": "", "citations": []}
    ans = ClaudeAnswerer(settings, client=fake_client([], payload)).answer("blood type?", hits)
    assert ans.text == NOT_FOUND


@pytest.mark.skipif(not os.environ.get("INTAKE_PG_DSN"), reason="INTAKE_PG_DSN not set")
def test_pgvector_store_matches_in_memory(corpus):
    sources = list(corpus[0].values())
    emb = HashingEmbedder()
    store = PgVectorStore(os.environ["INTAKE_PG_DSN"], emb.dim, table="rag_chunks_test")
    store.clear()
    pg, mem = RagIndex(emb, store), RagIndex(emb, InMemoryStore())
    n = pg.build(sources)
    mem.build(sources)
    assert store.count() == n
    for q, kw in [
        ("Does Northwind Health Plan require prior authorization for CPT 64483?", {}),
        ("fax number Tailspin", {"prefix": "policy:"}),
        ("member id", {"doc_id": first_form(corpus)}),
    ]:
        got = [h.chunk.chunk_id for h in pg.retrieve(q, k=3, **kw)]
        want = [h.chunk.chunk_id for h in mem.retrieve(q, k=3, **kw)]
        assert got == want
    hit = pg.retrieve("member id", k=1, doc_id=first_form(corpus))[0]
    assert corpus[0][hit.chunk.doc_id].text[hit.chunk.start : hit.chunk.end] == hit.chunk.text
    store.clear()


def test_make_embedder_honours_env(monkeypatch):
    from intake_agent.rag import make_embedder

    assert isinstance(make_embedder(), HashingEmbedder)  # conftest forces hashing
    monkeypatch.setenv("INTAKE_EMBEDDER", "bogus")
    with pytest.raises(ValueError):
        make_embedder()


def test_auto_falls_back_to_hashing_when_model_unavailable(monkeypatch):
    from intake_agent.rag import embed

    def boom(*a, **k):
        raise ImportError("no fastembed")

    monkeypatch.setenv("INTAKE_EMBEDDER", "auto")
    monkeypatch.setattr(embed, "FastEmbedder", boom)
    assert isinstance(embed.make_embedder(), HashingEmbedder)


def test_real_embedder_bridges_a_synonym(corpus):
    pytest.importorskip("fastembed")
    from intake_agent.rag import FastEmbedder

    idx = RagIndex(FastEmbedder(), InMemoryStore())
    idx.build(list(corpus[0].values()))
    hits = idx.retrieve(
        "How quickly does Contoso Care decide a rush request?", k=5, prefix="policy:"
    )
    assert any(h.chunk.doc_id == "policy:contoso-care" and "48 hours" in h.chunk.text for h in hits)
