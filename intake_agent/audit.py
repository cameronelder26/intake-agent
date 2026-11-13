"""Append-only, hash-chained JSONL audit log.

Every record carries who, what, when, the model and prompt hash that produced the
value, and the before/after value. Each record also stores the hash of the previous
record, so editing or deleting a line in the middle breaks `verify()`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _digest(prev_hash: str, record: dict[str, Any]) -> str:
    body = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev_hash + body).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        actor: str,
        action: str,
        doc_id: str | None = None,
        field: str | None = None,
        before: Any = None,
        after: Any = None,
        model: str | None = None,
        prompt_hash: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not actor:
            raise ValueError("audit records need an actor")
        # Open in append mode and hold an exclusive lock while reading the tail,
        # so two writers cannot fork the chain.
        with self.path.open("a+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            lines = fh.read().splitlines()
            last = json.loads(lines[-1]) if lines else None
            record = {
                "seq": (last["seq"] + 1) if last else 1,
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "actor": actor,
                "action": action,
                "doc_id": doc_id,
                "field": field,
                "before": before,
                "after": after,
                "model": model,
                "prompt_hash": prompt_hash,
                "detail": detail or {},
                "prev_hash": last["hash"] if last else GENESIS,
            }
            record["hash"] = _digest(record["prev_hash"], record)
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def verify(self) -> tuple[bool, int | None]:
        """Check the chain. Returns (ok, seq of first bad record or None)."""
        prev = GENESIS
        for rec in self.records():
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("prev_hash") != prev or _digest(prev, body) != rec.get("hash"):
                return False, rec.get("seq")
            prev = rec["hash"]
        return True, None
