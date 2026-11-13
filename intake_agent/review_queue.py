"""SQLite review queue and release gate.

Document lifecycle:

    extracted --(every field >= threshold)--> auto_approved --release--> released
        \\--(any field < threshold)--> pending_review --human--> approved --release--> released
                                                       \\--human--> rejected

`release()` is the only way a record leaves the system, and it refuses anything not in
auto_approved or approved. Every state change writes an audit record.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audit import AuditLog
from .schema import FIELDS, LIST_FIELDS, ExtractionResult, FieldValue, ScoredField
from .validators import normalize_value

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source_path  TEXT NOT NULL,
    status       TEXT NOT NULL,
    fields_json  TEXT NOT NULL,
    model        TEXT NOT NULL,
    prompt_hash  TEXT NOT NULL,
    cost_usd     REAL NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id           TEXT NOT NULL REFERENCES documents(doc_id),
    field            TEXT NOT NULL,
    value_json       TEXT,
    confidence       REAL NOT NULL,
    issues_json      TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'open',
    final_value_json TEXT,
    reviewer         TEXT,
    decided_at       TEXT,
    UNIQUE (doc_id, field)
);
"""

RELEASABLE = ("auto_approved", "approved")
_KEEP = object()  # sentinel: approve the field with its extracted value


class QueueError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_correction(field: str, text: str) -> FieldValue:
    """Turn CLI input into a field value. Lists are comma separated; '' or 'null' clears."""
    if field not in FIELDS:
        raise QueueError(f"unknown field {field!r}")
    if field in LIST_FIELDS:
        return normalize_value(field, [p for p in text.split(",") if p.strip()])
    if text.strip().lower() in ("", "null", "none"):
        return None
    return normalize_value(field, text)


class ReviewQueue:
    def __init__(self, db_path: Path, audit: AuditLog):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.audit = audit
        with closing(self._conn()) as conn, conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------------------------------------------------- intake

    def has(self, doc_id: str) -> bool:
        with closing(self._conn()) as conn:
            return (
                conn.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
                is not None
            )

    def add_document(
        self,
        doc_id: str,
        source_path: Path,
        scored: dict[str, ScoredField],
        result: ExtractionResult,
        threshold: float,
    ) -> str:
        flagged = [f for f in scored.values() if f.confidence < threshold]
        status = "pending_review" if flagged else "auto_approved"
        now = _now()
        fields_json = json.dumps({n: f.model_dump() for n, f in scored.items()})
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    str(source_path),
                    status,
                    fields_json,
                    result.model,
                    result.prompt_hash,
                    result.cost_usd,
                    now,
                    now,
                ),
            )
            conn.executemany(
                "INSERT INTO review_items (doc_id, field, value_json, confidence, issues_json)"
                " VALUES (?,?,?,?,?)",
                [
                    (doc_id, f.name, json.dumps(f.value), f.confidence, json.dumps(f.issues))
                    for f in flagged
                ],
            )
        common = {
            "actor": "system:pipeline",
            "doc_id": doc_id,
            "model": result.model,
            "prompt_hash": result.prompt_hash,
        }
        self.audit.append(
            action="extracted",
            after={n: f.value for n, f in scored.items()},
            detail={
                "confidence": {n: f.confidence for n, f in scored.items()},
                "used_document_block": result.used_document_block,
                "cost_usd": result.cost_usd,
            },
            **common,
        )
        self.audit.append(
            action=status,
            before=None,
            after=status,
            detail={"threshold": threshold, "flagged": [f.name for f in flagged]},
            **common,
        )
        return status

    # ---------------------------------------------------------- reads

    def document(self, doc_id: str) -> dict[str, Any]:
        with closing(self._conn()) as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if row is None:
            raise QueueError(f"no document {doc_id!r}")
        doc = dict(row)
        doc["fields"] = json.loads(doc.pop("fields_json"))
        return doc

    def items(self, doc_id: str, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM review_items WHERE doc_id=?"
        args: tuple = (doc_id,)
        if status:
            sql += " AND status=?"
            args += (status,)
        with closing(self._conn()) as conn:
            rows = conn.execute(sql + " ORDER BY id", args).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["value"] = json.loads(item.pop("value_json"))
            item["issues"] = json.loads(item.pop("issues_json"))
            fv = item.pop("final_value_json")
            item["final_value"] = json.loads(fv) if fv is not None else None
            out.append(item)
        return out

    def documents(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT doc_id, status, model, cost_usd, created_at FROM documents"
        args: tuple = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        with closing(self._conn()) as conn:
            return [dict(r) for r in conn.execute(sql + " ORDER BY doc_id", args).fetchall()]

    def pending(self) -> list[dict[str, Any]]:
        """Documents awaiting review, with their count of open fields."""
        with closing(self._conn()) as conn:
            rows = conn.execute(
                "SELECT d.doc_id, d.source_path, COUNT(i.id) AS open_fields,"
                " MIN(i.confidence) AS min_confidence"
                " FROM documents d JOIN review_items i ON i.doc_id = d.doc_id"
                " WHERE d.status='pending_review' AND i.status='open'"
                " GROUP BY d.doc_id ORDER BY d.doc_id"
            ).fetchall()
        return [dict(r) for r in rows]

    def final_fields(self, doc_id: str) -> dict[str, FieldValue]:
        doc = self.document(doc_id)
        values = {n: f["value"] for n, f in doc["fields"].items()}
        for item in self.items(doc_id):
            if item["status"] == "corrected":
                values[item["field"]] = item["final_value"]
        return values

    # ---------------------------------------------------------- decisions

    def _require_pending(self, doc_id: str) -> dict[str, Any]:
        doc = self.document(doc_id)
        if doc["status"] != "pending_review":
            raise QueueError(f"{doc_id} is {doc['status']}, not pending_review")
        return doc

    def _decide_item(
        self,
        doc_id: str,
        field: str,
        status: str,
        reviewer: str,
        final: FieldValue | object = _KEEP,
    ) -> None:
        if not reviewer:
            raise QueueError("a reviewer name is required")
        doc = self._require_pending(doc_id)
        open_items = {i["field"]: i for i in self.items(doc_id, "open")}
        if field not in open_items:
            raise QueueError(f"{doc_id}.{field} is not an open review item")
        before = open_items[field]["value"]
        if final is _KEEP:
            final = before
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE review_items SET status=?, final_value_json=?, reviewer=?, decided_at=?"
                " WHERE doc_id=? AND field=?",
                (status, json.dumps(final), reviewer, _now(), doc_id, field),
            )
        self.audit.append(
            actor=reviewer,
            action=f"field_{status}",
            doc_id=doc_id,
            field=field,
            before=before,
            after=final,
            model=doc["model"],
            prompt_hash=doc["prompt_hash"],
            detail={
                "confidence": open_items[field]["confidence"],
                "issues": open_items[field]["issues"],
            },
        )
        if len(open_items) == 1:
            self._set_status(doc, "approved", reviewer)

    def approve_field(self, doc_id: str, field: str, reviewer: str) -> None:
        self._decide_item(doc_id, field, "approved", reviewer)

    def correct_field(self, doc_id: str, field: str, value: FieldValue, reviewer: str) -> None:
        self._decide_item(doc_id, field, "corrected", reviewer, value)

    def approve_document(self, doc_id: str, reviewer: str) -> None:
        """Approve every remaining open field as extracted."""
        self._require_pending(doc_id)
        for item in self.items(doc_id, "open"):
            self._decide_item(doc_id, item["field"], "approved", reviewer)

    def reject_document(self, doc_id: str, reviewer: str, reason: str) -> None:
        if not reason:
            raise QueueError("a rejection reason is required")
        doc = self._require_pending(doc_id)
        self._set_status(doc, "rejected", reviewer, {"reason": reason})

    def _set_status(
        self, doc: dict[str, Any], status: str, actor: str, detail: dict[str, Any] | None = None
    ) -> None:
        with closing(self._conn()) as conn, conn:
            conn.execute(
                "UPDATE documents SET status=?, updated_at=? WHERE doc_id=?",
                (status, _now(), doc["doc_id"]),
            )
        self.audit.append(
            actor=actor,
            action=f"document_{status}",
            doc_id=doc["doc_id"],
            before=doc["status"],
            after=status,
            model=doc["model"],
            prompt_hash=doc["prompt_hash"],
            detail=detail,
        )

    # ---------------------------------------------------------- release gate

    def release(self, doc_id: str, actor: str = "system:release") -> dict[str, Any]:
        """Hand a record downstream. Refuses anything a human or the gate has not approved."""
        doc = self.document(doc_id)
        if doc["status"] not in RELEASABLE:
            raise QueueError(f"{doc_id} is {doc['status']}; only {RELEASABLE} can be released")
        record = {
            "doc_id": doc_id,
            "fields": self.final_fields(doc_id),
            "approval": doc["status"],
            "model": doc["model"],
            "prompt_hash": doc["prompt_hash"],
        }
        self._set_status(doc, "released", actor)
        return record
