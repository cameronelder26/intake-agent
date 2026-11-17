"""Checkpointed interrupt/resume, and the reviewer agent in front of the human queue."""

import pytest
from fastapi.testclient import TestClient

from intake_agent.api import create_app
from intake_agent.confidence import score_all
from intake_agent.extractor import MockExtractor
from intake_agent.graph import (
    build_graph,
    make_checkpointer,
    open_review_graph,
    pending_threads,
    resume_document,
    run_document,
)
from intake_agent.multiagent import MockReviewer, ModelReviewer, ReviewTools, make_reviewer
from intake_agent.schema import RawField


def _scan(dataset):
    return next(
        p
        for p in sorted(dataset.glob("*.pdf"))
        if not MockExtractor().extract(p).fields["patient_name"].value
    )


def _bad_npi_pdf(dataset, tmp_path):
    """A text document whose NPI fails its checksum (the mock keeps the printed value)."""
    from intake_agent.synth import generate

    out = tmp_path / "npi"
    generate(out, n=40, seed=3)
    for pdf in sorted(out.glob("*.pdf")):
        r = score_all(MockExtractor().extract(pdf).fields)
        if r["referring_npi"].issues and r["referring_npi"].value and not r["patient_name"].issues:
            return pdf
    pytest.skip("no bad-NPI document in the synthetic set")


# ------------------------------------------------------------ interrupt and resume


def test_flagged_document_pauses_then_resumes(dataset, settings, state):
    _, queue = state
    cp = make_checkpointer(None)
    scan = _scan(dataset)
    out = run_document(scan, MockExtractor(), queue, settings, checkpointer=cp)
    assert out["__interrupt__"]
    assert queue.document(out["doc_id"])["status"] == "pending_review"
    # No audit "graph_routed" yet: the graph has not reached its last node.
    assert not [r for r in queue.audit.records() if r["action"] == "graph_routed"]

    graph = build_graph(MockExtractor(), queue, settings, checkpointer=cp)
    pend = pending_threads(graph)
    assert [p["thread_id"] for p in pend] == [out["doc_id"]]
    assert {f["field"] for f in pend[0]["fields"]} == set(out["flagged"])

    final = resume_document(graph, out["doc_id"], {"reviewer": "dana", "approve_all": True})
    assert final["status"] == "approved"
    assert queue.document(out["doc_id"])["status"] == "approved"
    assert [r for r in queue.audit.records() if r["action"] == "graph_routed"]
    assert queue.audit.verify()[0]
    assert pending_threads(graph) == []


def test_resume_with_correction_and_unknown_thread(dataset, settings, state):
    _, queue = state
    cp = make_checkpointer(None)
    scan = _scan(dataset)
    out = run_document(scan, MockExtractor(), queue, settings, checkpointer=cp)
    graph = build_graph(MockExtractor(), queue, settings, checkpointer=cp)
    first, *rest = out["flagged"]
    resume_document(graph, out["doc_id"], {"reviewer": "dana", "decisions": {first: "approve"}})
    # One field decided, the rest still open: the document stays in the queue.
    assert queue.document(out["doc_id"])["status"] == "pending_review"
    assert len(queue.items(out["doc_id"], "open")) == len(rest)
    with pytest.raises(KeyError):
        resume_document(graph, "no-such-thread", {"reviewer": "dana"})


def test_sqlite_checkpoint_survives_a_new_graph(dataset, settings, state):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    _, queue = state
    out = run_document(_scan(dataset), MockExtractor(), queue, settings)
    assert (settings.state_dir / "checkpoints.sqlite").exists()
    fresh = open_review_graph(settings)  # new graph, new connection, same files
    assert [p["thread_id"] for p in pending_threads(fresh)] == [out["doc_id"]]
    resume_document(fresh, out["doc_id"], {"reviewer": "dana", "approve_all": True})
    assert queue.document(out["doc_id"])["status"] == "approved"


def test_api_pending_and_resume(dataset, settings, state):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    _, queue = state
    out = run_document(_scan(dataset), MockExtractor(), queue, settings)
    client = TestClient(create_app(lambda: settings, mock=True))
    pend = client.get("/review/pending").json()
    assert pend["count"] == 1 and pend["items"][0]["thread_id"] == out["doc_id"]
    bad = client.post(
        f"/review/{out['doc_id']}/resume", json={"reviewer": "dana", "decisions": {"payer": "nope"}}
    )
    assert bad.status_code == 422
    ok = client.post(
        f"/review/{out['doc_id']}/resume", json={"reviewer": "dana", "approve_all": True}
    )
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    assert client.get("/review/pending").json()["count"] == 0
    assert client.post("/review/ghost/resume", json={"reviewer": "dana"}).status_code == 404


def test_cli_graph_review_lists_and_resumes(dataset, settings, state, monkeypatch):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from typer.testing import CliRunner

    from intake_agent.cli import app

    _, queue = state
    out = run_document(_scan(dataset), MockExtractor(), queue, settings)
    monkeypatch.setenv("INTAKE_STATE_DIR", str(settings.state_dir))
    r = CliRunner().invoke(app, ["graph-review"])
    assert out["doc_id"] in r.output
    r = CliRunner().invoke(
        app, ["graph-review", out["doc_id"], "--reviewer", "dana", "--approve-all"]
    )
    assert "status=approved" in r.output, r.output


# ------------------------------------------------------------ the reviewer agent


def _scored(**kw):
    raw = {
        "referring_npi": RawField(value="1234567890", model_confidence=0.7),  # bad checksum
        "payer": RawField(value="Contoso Care", model_confidence=0.7),
        "patient_name": RawField(value="Avery Samplesen", model_confidence=0.7),
    }
    raw.update(kw)
    return score_all(raw)


TEXT = "Patient: Avery Samplesen\nPayer: Contoso Care\nNPI: 1234567890\n"


def test_reviewer_catches_seeded_bad_fields():
    scored = _scored(payer=RawField(value="Fabrikam Mutual", model_confidence=0.7))
    verdicts = {v.field: v for v in MockReviewer().review(scored, list(scored), TEXT)}
    assert verdicts["referring_npi"].decision == "escalate"
    assert "checksum" in verdicts["referring_npi"].reason
    # Valid on its face but not written anywhere in the source: a hallucination.
    assert verdicts["payer"].decision == "escalate"
    assert "not found" in verdicts["payer"].reason


def test_reviewer_lets_good_fields_through():
    scored = _scored()
    verdicts = {v.field: v for v in MockReviewer().review(scored, ["payer", "patient_name"], TEXT)}
    assert all(v.decision == "clear" for v in verdicts.values())


def test_tools_wrap_the_validators():
    tools = ReviewTools(_scored(), TEXT)
    assert tools.call("validate_field", {"field": "referring_npi"})["valid"] is False
    assert tools.call("lookup_source_span", {"field": "payer"})["quote"] == "Contoso Care"
    assert "error" in tools.call("lookup_source_span", {"field": "bogus"})
    assert "error" in tools.call("rm_rf", {"field": "payer"})


def test_graph_reviewer_clears_good_and_escalates_bad(dataset, settings, state, tmp_path):
    _, queue = state
    # Bad NPI: still reaches a human, with the reviewer's note attached.
    pdf = _bad_npi_pdf(dataset, tmp_path)
    cp = make_checkpointer(None)
    out = run_document(
        pdf, MockExtractor(), queue, settings, reviewer=MockReviewer(), checkpointer=cp
    )
    assert "referring_npi" in out["flagged"]
    payload = out["__interrupt__"][0].value
    npi = next(f for f in payload["fields"] if f["field"] == "referring_npi")
    assert "checksum" in npi["reviewer_note"]
    assert any(
        r["action"] == "reviewer_verdicts" and r["actor"].startswith("agent:")
        for r in queue.audit.records()
    )


def test_graph_reviewer_clears_low_confidence_but_correct_fields(dataset, settings, state):
    _, queue = state

    # Every field of a 0.7-confidence document is right and written in the source.
    class Hedging(MockExtractor):
        def extract(self, path):
            res = super().extract(path)
            for f in res.fields.values():
                if f.value not in (None, []):
                    f.model_confidence = 0.7
            return res

    pdf = next(
        p
        for p in sorted(dataset.glob("*.pdf"))
        if not any(i for i in score_all(MockExtractor().extract(p).fields).values() if i.issues)
    )
    without = run_document(
        pdf, Hedging(), queue, settings, doc_id="a", checkpointer=make_checkpointer(None)
    )
    assert without["route"] == "human_review"
    with_r = run_document(
        pdf,
        Hedging(),
        queue,
        settings,
        doc_id="b",
        reviewer=MockReviewer(),
        checkpointer=make_checkpointer(None),
    )
    assert with_r["route"] == "auto_accept"
    assert all(v["decision"] == "clear" for v in with_r["verdicts"])
    assert queue.document("b")["status"] == "auto_approved"


def test_make_reviewer_is_mock_without_a_key(settings, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_reviewer(None, settings), MockReviewer)


def test_model_reviewer_tool_loop_with_a_fake_client(settings):
    """The real-model path, driven by a scripted client: tool call, then verdict."""
    from types import SimpleNamespace as NS

    script = [
        NS(content=[NS(type="tool_use", id="1", name="validate_field", input={"field": "payer"})]),
        NS(
            content=[
                NS(
                    type="tool_use",
                    id="2",
                    name="submit_verdict",
                    input={"field": "payer", "decision": "clear", "reason": "valid"},
                )
            ]
        ),
    ]
    client = NS(messages=NS(create=lambda **kw: script.pop(0)))
    rv = ModelReviewer(settings, client=client)
    (v,) = rv.review(_scored(), ["payer"], TEXT)
    assert (v.field, v.decision) == ("payer", "clear")


def test_tracing_is_a_noop_without_a_key(monkeypatch):
    from intake_agent import tracing

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert tracing.configure() is False
    import os

    assert os.environ["LANGSMITH_TRACING"] == "false"
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "x")
    assert tracing.configure() is True


def test_langsmith_script_offline_and_dataset_shape(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "langsmith_dataset", Path(__file__).parent.parent / "evals" / "langsmith_dataset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.chdir(Path(__file__).parent.parent)
    assert mod.main([]) == 0  # no key: does nothing
    assert "not set" in capsys.readouterr().out
    examples = [mod.example_for(t) for t in mod.load_tasks()]
    assert len(examples) == 32
    assert {e["inputs"]["kind"] for e in examples} == {"extract", "rag"}
