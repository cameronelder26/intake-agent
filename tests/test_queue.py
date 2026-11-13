from datetime import date
from pathlib import Path

import pytest

from intake_agent.confidence import score_all
from intake_agent.review_queue import QueueError, parse_correction
from intake_agent.schema import FIELDS, ExtractionResult, RawField

TODAY = date(2026, 11, 10)

GOOD = {
    "document_type": "referral",
    "patient_name": "Avery Testwell",
    "patient_dob": "1961-03-04",
    "member_id": "NWH123456789",
    "payer": "Northwind Health Plan",
    "referring_provider": "Riley Stubbins",
    "referring_npi": "1234567893",
    "icd10_codes": ["M17.11"],
    "cpt_codes": ["73721"],
    "requested_service": "MRI right knee without contrast",
    "urgency": "routine",
    "request_date": "2026-06-07",
}


def make_result(conf: float = 0.97, **overrides) -> ExtractionResult:
    values = {**GOOD, **overrides}
    return ExtractionResult(
        fields={n: RawField(value=values[n], model_confidence=conf) for n in FIELDS},
        model="test-model",
        prompt_hash="abc123",
    )


def add(queue, doc_id: str, result: ExtractionResult, threshold: float = 0.85) -> str:
    return queue.add_document(
        doc_id, Path(f"{doc_id}.pdf"), score_all(result.fields, TODAY), result, threshold
    )


def test_all_fields_above_threshold_auto_approves(state):
    _, queue = state
    assert add(queue, "d1", make_result()) == "auto_approved"
    assert queue.items("d1") == []


def test_low_field_goes_to_review(state):
    _, queue = state
    assert add(queue, "d1", make_result(referring_npi="1234567890")) == "pending_review"
    items = queue.items("d1")
    assert [i["field"] for i in items] == ["referring_npi"]
    assert items[0]["issues"] == ["NPI fails checksum"]
    assert queue.pending()[0]["doc_id"] == "d1"


def test_pending_document_cannot_be_released(state):
    _, queue = state
    add(queue, "d1", make_result(member_id=None))
    with pytest.raises(QueueError, match="pending_review"):
        queue.release("d1")


def test_correction_then_release(state):
    _, queue = state
    add(queue, "d1", make_result(member_id=None))
    queue.correct_field("d1", "member_id", "NWH999999999", reviewer="alice")
    assert queue.document("d1")["status"] == "approved"
    record = queue.release("d1")
    assert record["fields"]["member_id"] == "NWH999999999"
    assert record["approval"] == "approved"
    assert queue.document("d1")["status"] == "released"
    with pytest.raises(QueueError):
        queue.release("d1")  # already released, cannot go twice


def test_document_stays_pending_until_every_field_decided(state):
    _, queue = state
    add(queue, "d1", make_result(member_id=None, referring_npi="1234567890"))
    queue.approve_field("d1", "referring_npi", reviewer="alice")
    assert queue.document("d1")["status"] == "pending_review"
    queue.approve_document("d1", reviewer="alice")
    assert queue.document("d1")["status"] == "approved"


def test_rejected_never_released(state):
    _, queue = state
    add(queue, "d1", make_result(conf=0.5))
    with pytest.raises(QueueError, match="reason"):
        queue.reject_document("d1", reviewer="alice", reason="")
    queue.reject_document("d1", reviewer="alice", reason="wrong patient")
    with pytest.raises(QueueError):
        queue.release("d1")
    with pytest.raises(QueueError):
        queue.approve_document("d1", reviewer="alice")


def test_reviewer_required_and_unknown_field(state):
    _, queue = state
    add(queue, "d1", make_result(member_id=None))
    with pytest.raises(QueueError, match="reviewer"):
        queue.approve_field("d1", "member_id", reviewer="")
    with pytest.raises(QueueError, match="not an open review item"):
        queue.approve_field("d1", "payer", reviewer="alice")
    with pytest.raises(QueueError, match="unknown field"):
        parse_correction("favorite_color", "blue")


def test_parse_correction():
    assert parse_correction("icd10_codes", "m17.11, Z12.11") == ["M17.11", "Z12.11"]
    assert parse_correction("member_id", "null") is None
    assert parse_correction("patient_dob", "03/04/1961") == "1961-03-04"
