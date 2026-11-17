"""LangGraph orchestration: the confidence gate picks the branch."""

import pytest

from intake_agent.extractor import MockExtractor
from intake_agent.graph import KillSwitchEngaged, find_citation, run_document


def _by_route(dataset, settings, state):
    """Run every text-layer document and return {route: [doc ids]}."""
    _, queue = state
    routes: dict[str, list[str]] = {}
    for pdf in sorted(dataset.glob("*.pdf")):
        out = run_document(pdf, MockExtractor(), queue, settings)
        routes.setdefault(out["route"], []).append(out["doc_id"])
    return routes


def test_low_confidence_routes_to_human_review(dataset, settings, state):
    # The scanned document has no text layer, so every field comes back at 0.0.
    scan = next(
        p
        for p in sorted(dataset.glob("*.pdf"))
        if not MockExtractor().extract(p).fields["patient_name"].value
    )
    _, queue = state
    out = run_document(scan, MockExtractor(), queue, settings)
    assert out["route"] == "human_review"
    assert out["status"] == "pending_review"
    assert out["flagged"]
    assert queue.document(out["doc_id"])["status"] == "pending_review"


def test_clean_document_is_auto_accepted(dataset, settings, state):
    routes = _by_route(dataset, settings, state)
    assert routes.get("auto_accept"), routes
    assert routes.get("human_review"), routes


def test_audit_records_route_and_chain_stays_intact(dataset, settings, state):
    audit, queue = state
    pdf = sorted(dataset.glob("*.pdf"))[0]
    out = run_document(pdf, MockExtractor(), queue, settings)
    routed = [r for r in audit.records() if r["action"] == "graph_routed"]
    assert routed[-1]["doc_id"] == out["doc_id"]
    assert audit.verify()[0]


def test_graph_halts_on_kill_switch(dataset, settings, state, monkeypatch):
    monkeypatch.setenv("INTAKE_KILL_SWITCH", "1")
    _, queue = state
    with pytest.raises(KillSwitchEngaged):
        run_document(sorted(dataset.glob("*.pdf"))[0], MockExtractor(), queue, settings)
    assert not queue.documents()


def test_find_citation():
    text = "Name: Jane Roe\nCPT: 64483, 64484"
    assert find_citation("jane roe", text) == {"start": 6, "end": 14, "quote": "Jane Roe"}
    assert find_citation(["64483", "64484"], text)["quote"] == "64483, 64484"
    assert find_citation("absent", text) is None
    assert find_citation(None, text) is None
