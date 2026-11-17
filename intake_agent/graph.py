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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import killswitch, tracing
from .audit import AuditLog
from .confidence import needs_review, score_all
from .config import Settings
from .extractor import ExtractionError, Extractor
from .pdftext import collapse_fragments, extract_raw_text
from .review_queue import ReviewQueue, parse_correction
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
    verdicts: list[dict[str, str]]  # reviewer agent decisions, if one is wired in
    notes: dict[str, str]  # reviewer reason per escalated field
    decision: dict[str, Any]  # what the human decided at the interrupt


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


# Our own types that go into checkpoints. Listing them keeps LangGraph's msgpack
# deserializer strict about everything else.
_SERDE_TYPES = [
    ("intake_agent.schema", "ExtractionResult"),
    ("intake_agent.schema", "ScoredField"),
    ("intake_agent.schema", "RawField"),
]


def _serde():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=_SERDE_TYPES)


def make_checkpointer(settings: Settings | None = None):
    """SqliteSaver (survives restarts) if its package is installed, else MemorySaver.

    Pass None for settings to get an in-memory saver without touching disk.
    """
    if settings is not None:
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            settings.state_dir.mkdir(parents=True, exist_ok=True)
            path = settings.state_dir / "checkpoints.sqlite"
            return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=_serde())
        except ImportError:
            pass
    return MemorySaver(serde=_serde())


def build_graph(
    extractor: Extractor,
    queue: ReviewQueue,
    settings: Settings,
    audit: AuditLog | None = None,
    reviewer: Any = None,
    checkpointer: Any = None,
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

    def review_agent(state: IntakeState) -> IntakeState:
        """Optional second agent: challenge flagged fields before they reach a human."""
        if reviewer is None or not state["flagged"]:
            return {"verdicts": [], "notes": {}}
        verdicts = reviewer.review(state["scored"], state["flagged"], state["source_text"])
        scored = dict(state["scored"])
        notes = {}
        for v in verdicts:
            if v.decision == "clear":
                # Treated as checked: the stored confidence is lifted to the threshold.
                scored[v.field] = scored[v.field].model_copy(
                    update={"confidence": max(scored[v.field].confidence, settings.threshold)}
                )
            else:
                notes[v.field] = v.reason
        audit.append(
            actor=f"agent:{reviewer.name}",
            action="reviewer_verdicts",
            doc_id=state["doc_id"],
            detail={"verdicts": [v.as_dict() for v in verdicts]},
        )
        return {
            "scored": scored,
            "flagged": needs_review(scored, settings.threshold),
            "verdicts": [v.as_dict() for v in verdicts],
            "notes": notes,
        }

    def _store(route: str):
        def node(state: IntakeState) -> IntakeState:
            status = queue.add_document(
                state["doc_id"], state["path"], state["scored"], state["result"], settings.threshold
            )
            return {"route": route, "status": status}

        return node

    def human_review(state: IntakeState) -> IntakeState:
        """Pause for a person. Resume with Command(resume={...}); see apply_decision."""
        items = [
            {
                "field": n,
                "value": state["scored"][n].value,
                "confidence": state["scored"][n].confidence,
                "issues": state["scored"][n].issues,
                "citation": state["citations"].get(n),
                "reviewer_note": state.get("notes", {}).get(n),
            }
            for n in state["flagged"]
        ]
        decision = interrupt({"doc_id": state["doc_id"], "fields": items})
        apply_decision(queue, state["doc_id"], decision)
        return {"decision": decision, "status": queue.document(state["doc_id"])["status"]}

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
    g.add_node("reviewer", review_agent)
    g.add_node("queue_review", _store("human_review"))
    g.add_node("human_review", human_review)
    g.add_node("audit", record)
    g.add_edge(START, "extract")
    g.add_edge("extract", "validate")
    g.add_edge("validate", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route_after_gate,
        {"auto_accept": "auto_accept", "human_review": "queue_review"},
    )
    g.add_edge("auto_accept", "audit")
    g.add_edge("queue_review", "human_review")
    g.add_edge("human_review", "audit")
    g.add_edge("audit", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def apply_decision(queue: ReviewQueue, doc_id: str, decision: dict[str, Any]) -> None:
    """Apply a reviewer's resume payload to the queue.

    {"reviewer": "dana", "approve_all": true}
    {"reviewer": "dana", "decisions": {"referring_npi": {"correct": "1234567893"},
                                       "payer": "approve"}}
    {"reviewer": "dana", "reject": "wrong patient"}

    Fields not mentioned stay open, so the document stays in the queue.
    """
    reviewer = decision.get("reviewer") or ""
    if decision.get("reject"):
        queue.reject_document(doc_id, reviewer, decision["reject"])
        return
    if decision.get("approve_all"):
        queue.approve_document(doc_id, reviewer)
        return
    for field, d in (decision.get("decisions") or {}).items():
        if d == "approve":
            queue.approve_field(doc_id, field, reviewer)
        elif isinstance(d, dict) and "correct" in d:
            value = d["correct"]
            if isinstance(value, str):
                value = parse_correction(field, value)
            queue.correct_field(doc_id, field, value, reviewer)
        else:
            raise ValueError(f"decision for {field!r} must be 'approve' or {{'correct': value}}")


def thread_config(thread_id: str) -> dict[str, Any]:
    """Per-thread config. The name, tags and metadata show up on a LangSmith trace if
    tracing is on (see tracing.py) and are ignored otherwise."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "intake-document",
        "tags": ["intake-agent"],
        "metadata": {"doc_id": thread_id},
    }


def open_review_graph(settings: Settings, mock: bool = True):
    """The graph bound to the same on-disk checkpoints and queue, for listing and resuming.

    Resume only runs nodes after the interrupt, so the extractor here is never called.
    """
    from .extractor import make_extractor
    from .pipeline import open_state

    _, queue = open_state(settings)
    return build_graph(
        make_extractor(mock, settings), queue, settings, checkpointer=make_checkpointer(settings)
    )


def run_document(
    path: Path,
    extractor: Extractor,
    queue: ReviewQueue,
    settings: Settings,
    doc_id: str | None = None,
    reviewer: Any = None,
    checkpointer: Any = None,
) -> IntakeState:
    """Run one document. A flagged one pauses at the human_review interrupt.

    The returned state has route and status as before; if the graph paused it also
    has "__interrupt__". The thread id is the doc id.
    """
    tracing.configure()
    doc_id = doc_id or path.stem
    cp = checkpointer or make_checkpointer(settings)
    graph = build_graph(extractor, queue, settings, reviewer=reviewer, checkpointer=cp)
    return graph.invoke({"path": path, "doc_id": doc_id}, thread_config(doc_id))


def pending_threads(graph) -> list[dict[str, Any]]:
    """Threads paused at an interrupt, with the payload the reviewer should see."""
    seen: dict[str, None] = {}
    for tup in graph.checkpointer.list(None):
        seen[tup.config["configurable"]["thread_id"]] = None
    out = []
    for tid in seen:
        for task in graph.get_state(thread_config(tid)).tasks:
            for itr in task.interrupts:
                out.append({"thread_id": tid, **itr.value})
    return out


def resume_document(graph, thread_id: str, decision: dict[str, Any]) -> IntakeState:
    snap = graph.get_state(thread_config(thread_id))
    interrupts = [i for t in snap.tasks for i in t.interrupts]
    if not interrupts:
        raise KeyError(f"no pending review for thread {thread_id!r}")
    # Check the payload before resuming: a resume value that raises inside the node is
    # kept by the checkpointer and would be replayed on the next attempt.
    if not decision.get("reviewer"):
        raise ValueError("a reviewer name is required")
    open_fields = {f["field"] for f in interrupts[0].value["fields"]}
    for field, d in (decision.get("decisions") or {}).items():
        if field not in open_fields:
            raise ValueError(f"{field!r} is not an open field of {thread_id}")
        if d != "approve" and not (isinstance(d, dict) and "correct" in d):
            raise ValueError(f"decision for {field!r} must be 'approve' or {{'correct': value}}")
    return graph.invoke(Command(resume=decision), thread_config(thread_id))
