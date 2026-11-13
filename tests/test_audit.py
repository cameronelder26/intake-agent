import json
from datetime import date
from pathlib import Path

import pytest

from intake_agent.audit import AuditLog
from intake_agent.confidence import score_all

from .test_queue import make_result

TODAY = date(2026, 11, 10)


def test_append_only_chain(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    first = log.append(actor="alice", action="a", before=1, after=2)
    second = log.append(actor="bob", action="b")
    assert second["prev_hash"] == first["hash"]
    assert [r["seq"] for r in log.records()] == [1, 2]
    assert log.verify() == (True, None)


def test_edit_in_the_middle_is_detected(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.append(actor="alice", action="set", after=i)
    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["after"] = 999
    lines[1] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    assert log.verify() == (False, 2)


def test_deleted_line_is_detected(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.append(actor="alice", action="set", after=i)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert log.verify() == (False, 3)


def test_actor_required(tmp_path: Path):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "a.jsonl").append(actor="", action="x")


def test_every_decision_is_logged_with_before_after(state):
    audit, queue = state
    result = make_result(member_id=None, referring_npi="1234567890")
    queue.add_document("d1", Path("d1.pdf"), score_all(result.fields, TODAY), result, 0.85)
    queue.correct_field("d1", "member_id", "NWH111111111", reviewer="alice")
    queue.approve_field("d1", "referring_npi", reviewer="bob")
    queue.release("d1")

    recs = audit.records()
    actions = [r["action"] for r in recs]
    assert actions == [
        "extracted",
        "pending_review",
        "field_corrected",
        "field_approved",
        "document_approved",
        "document_released",
    ]
    corrected = recs[2]
    assert corrected["actor"] == "alice"
    assert corrected["field"] == "member_id"
    assert corrected["before"] is None
    assert corrected["after"] == "NWH111111111"
    assert all(r["model"] == "test-model" and r["prompt_hash"] == "abc123" for r in recs)
    assert all(r["ts"] for r in recs)
    assert audit.verify() == (True, None)
