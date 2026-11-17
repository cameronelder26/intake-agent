"""LangGraph orchestration of one document: extract, validate, gate, queue, audit.

    extract -> validate -> (confidence gate) -> auto_accept  -> audit
                                             -> human_review -> audit

The nodes are thin wrappers over the existing modules. The extractor reads the
document, confidence.score_all applies the validators, the gate compares every
field to the threshold, and the review queue stores the result and writes the
audit records it always wrote. The last node adds one audit record that says which
branch the document took and why.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import killswitch
from .audit import AuditLog
from .confidence import needs_review, score_all
from .config import Settings
from .extractor import ExtractionError, Extractor
from .pdftext import collapse_fragments, extract_raw_text
from .review_queue import ReviewQueue
from .schema import ExtractionResult, ScoredField


class KillSwitchEngaged(RuntimeError):
    pass


class IntakeState(TypedDict, total=False):
    path: Path
    doc_id: str
    result: ExtractionResult
    scored: dict[str, ScoredField]
    flagged: list[str]
    route: str  # "auto_accept" or "human_review"
    status: str  # status the review queue assigned
    source_text: str
    citations: dict[str, dict[str, Any] | None]


def _variants(value: Any) -> list[str]:
    """Spellings a normalized value may have had in the source."""
    text = str(value)
    out = [text]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        y, m, d = text.split("-")
        out += [f"{m}/{d}/{y}", f"{int(m)}/{int(d)}/{y}"]
    return out


def _span(needle: str, lowered: str) -> tuple[int, int] | None:
    i = lowered.find(needle.lower())
    if i >= 0:
        return i, i + len(needle)
    # Names are normalized ("STUBBINS, JORDAN" -> "Jordan Stubbins"): accept every word
    # of the value found close together.
    words = [w for w in re.split(r"\W+", needle.lower()) if w]
    if len(words) < 2:
        return None
    found = [(lowered.find(w), w) for w in words]
    if any(i < 0 for i, _ in found):
        return None
    start = min(i for i, _ in found)
    end = max(i + len(w) for i, w in found)
    return (start, end) if end - start <= 60 else None


def find_citation(value: Any, text: str) -> dict[str, Any] | None:
    """Locate a field value in the source text. None if it is not written there."""
    lowered = text.lower()
    spans = []
    for n in value if isinstance(value, list) else [value]:
        if not n:
            continue
        for variant in _variants(n):
            span = _span(variant, lowered)
            if span:
                spans.append(span)
                break
    if not spans:
        return None
    start, end = min(s for s, _ in spans), max(e for _, e in spans)
    return {"start": start, "end": end, "quote": text[start:end]}


def route_after_gate(state: IntakeState) -> str:
    return "human_review" if state["flagged"] else "auto_accept"


def build_graph(
    extractor: Extractor, queue: ReviewQueue, settings: Settings, audit: AuditLog | None = None
):
    audit = audit or queue.audit

    def extract(state: IntakeState) -> IntakeState:
        reason = killswitch.engaged(settings)
        if reason:
            audit.append(
                actor="system:graph",
                action="halted_by_kill_switch",
                doc_id=state["doc_id"],
                detail={"reason": reason},
            )
            raise KillSwitchEngaged(reason)
        try:
            result = extractor.extract(state["path"])
        except ExtractionError as exc:
            audit.append(
                actor="system:graph",
                action="extraction_failed",
                doc_id=state["doc_id"],
                model=extractor.model,
                prompt_hash=extractor.prompt_hash,
                detail={"error": str(exc)},
            )
            raise
        return {"result": result}

    def validate(state: IntakeState) -> IntakeState:
        scored = score_all(state["result"].fields)
        try:
            text = collapse_fragments(extract_raw_text(state["path"]))
        except Exception:  # unreadable PDF: no citations, the confidence still gates
            text = ""
        citations = {name: find_citation(f.value, text) for name, f in scored.items()}
        return {
            "scored": scored,
            "flagged": needs_review(scored, settings.threshold),
            "source_text": text,
            "citations": citations,
        }

    def _store(route: str):
        def node(state: IntakeState) -> IntakeState:
            status = queue.add_document(
                state["doc_id"], state["path"], state["scored"], state["result"], settings.threshold
            )
            return {"route": route, "status": status}

        return node

    def record(state: IntakeState) -> IntakeState:
        audit.append(
            actor="system:graph",
            action="graph_routed",
            doc_id=state["doc_id"],
            model=state["result"].model,
            prompt_hash=state["result"].prompt_hash,
            detail={
                "route": state["route"],
                "status": state["status"],
                "threshold": settings.threshold,
                "flagged_fields": state["flagged"],
            },
        )
        return {}

    g = StateGraph(IntakeState)
    g.add_node("extract", extract)
    g.add_node("validate", validate)
    g.add_node("auto_accept", _store("auto_accept"))
    g.add_node("human_review", _store("human_review"))
    g.add_node("audit", record)
    g.add_edge(START, "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges(
        "validate",
        route_after_gate,
        {"auto_accept": "auto_accept", "human_review": "human_review"},
    )
    g.add_edge("auto_accept", "audit")
    g.add_edge("human_review", "audit")
    g.add_edge("audit", END)
    return g.compile()


def run_document(
    path: Path,
    extractor: Extractor,
    queue: ReviewQueue,
    settings: Settings,
    doc_id: str | None = None,
) -> IntakeState:
    graph = build_graph(extractor, queue, settings)
    return graph.invoke({"path": path, "doc_id": doc_id or path.stem})
