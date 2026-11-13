"""Batch pipeline: kill-switch check, extract, score, gate, queue."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import killswitch
from .audit import AuditLog
from .confidence import score_all
from .config import Settings
from .extractor import ExtractionError, Extractor
from .review_queue import ReviewQueue


@dataclass
class BatchReport:
    processed: list[str] = field(default_factory=list)
    auto_approved: list[str] = field(default_factory=list)
    pending_review: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    halted_reason: str | None = None
    cost_usd: float = 0.0


def open_state(settings: Settings) -> tuple[AuditLog, ReviewQueue]:
    audit = AuditLog(settings.audit_path)
    return audit, ReviewQueue(settings.db_path, audit)


def process_batch(
    paths: Iterable[Path],
    extractor: Extractor,
    queue: ReviewQueue,
    settings: Settings,
    today: date | None = None,
) -> BatchReport:
    report = BatchReport()
    audit = queue.audit
    for path in paths:
        # Checked per document so an operator can stop a batch mid-run.
        reason = killswitch.engaged(settings)
        if reason:
            report.halted_reason = reason
            audit.append(
                actor="system:pipeline",
                action="halted_by_kill_switch",
                doc_id=path.stem,
                detail={"reason": reason},
            )
            break
        doc_id = path.stem
        if queue.has(doc_id):
            report.skipped.append(doc_id)
            continue
        try:
            result = extractor.extract(path)
        except ExtractionError as exc:
            report.failed.append(doc_id)
            audit.append(
                actor="system:pipeline",
                action="extraction_failed",
                doc_id=doc_id,
                model=extractor.model,
                prompt_hash=extractor.prompt_hash,
                detail={"error": str(exc)},
            )
            continue
        scored = score_all(result.fields, today)
        status = queue.add_document(doc_id, path, scored, result, settings.threshold)
        report.processed.append(doc_id)
        report.cost_usd += result.cost_usd
        (report.auto_approved if status == "auto_approved" else report.pending_review).append(
            doc_id
        )
    return report
