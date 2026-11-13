"""Extractor tests. The Claude path runs against a fake HTTP transport, no network."""

import json
from pathlib import Path

import anthropic
import httpx2 as httpx
import pytest

from intake_agent.eval import run_eval, values_match
from intake_agent.extractor import PROMPT_HASH, ClaudeExtractor, ExtractionError, MockExtractor
from intake_agent.pdftext import collapse_fragments, extract_raw_text, has_usable_text
from intake_agent.schema import FIELDS


def truth(dataset: Path, template: str) -> tuple[Path, dict]:
    for f in sorted(dataset.glob("*.json")):
        data = json.loads(f.read_text())
        if data["meta"]["template"] == template:
            return f.with_suffix(".pdf"), data
    raise LookupError(template)


def fake_client(captured: list[dict], payload: dict | None = None, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if status != 200:
            return httpx.Response(
                status,
                json={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "bad"},
                },
            )
        fields = payload or {
            n: {"value": [] if n.endswith("codes") else None, "confidence": 0.5} for n in FIELDS
        }
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": json.dumps(fields)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1000, "output_tokens": 300},
            },
        )

    return anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_text_document_sends_text_block(dataset, settings):
    pdf, gold = truth(dataset, "form_grid")
    payload = {n: {"value": v, "confidence": 0.93} for n, v in gold["fields"].items()}
    captured: list[dict] = []
    ex = ClaudeExtractor(settings, client=fake_client(captured, payload))
    result = ex.extract(pdf)

    req = captured[0]
    assert req["model"] == "claude-haiku-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["output_config"]["effort"] == "low"
    assert [b["type"] for b in req["messages"][0]["content"]] == ["text"]
    assert result.used_document_block is False
    assert result.prompt_hash == PROMPT_HASH
    assert result.fields["member_id"].value == gold["fields"]["member_id"]
    assert result.cost_usd == pytest.approx((1000 * 0.10 + 300 * 0.50) / 1e6)


def test_scanned_document_falls_back_to_pdf_block(dataset, settings):
    pdf, _ = truth(dataset, "scanned")
    assert not has_usable_text(extract_raw_text(pdf))
    captured: list[dict] = []
    result = ClaudeExtractor(settings, client=fake_client(captured)).extract(pdf)
    blocks = captured[0]["messages"][0]["content"]
    assert blocks[0]["type"] == "document"
    assert blocks[0]["source"]["media_type"] == "application/pdf"
    assert result.used_document_block is True


def test_api_error_becomes_extraction_error(dataset, settings):
    pdf, _ = truth(dataset, "form_grid")
    with pytest.raises(ExtractionError):
        ClaudeExtractor(settings, client=fake_client([], status=400)).extract(pdf)


def test_collapse_fragments():
    assert collapse_fragments("Pt name:\nA\nv\n \nB\nDOB:") == "Pt name:\nAv B\nDOB:"


def test_mock_reads_handwritten_but_reports_low_confidence(dataset):
    pdf, gold = truth(dataset, "fax_handwritten")
    result = MockExtractor().extract(pdf)
    assert values_match(
        "patient_name", result.fields["patient_name"].value, gold["fields"]["patient_name"]
    )
    assert result.fields["patient_name"].model_confidence < 0.85


def test_mock_eval_runs_end_to_end(dataset, settings):
    res = run_eval(dataset, mock=True, settings=settings)
    assert res.n_docs == 8
    assert res.total_cost == 0.0
    # Scans have no text layer, so the mock must send every scan to review.
    assert res.per_template["scanned"][2] == res.per_template["scanned"][0]
    assert res.auto_docs_with_error == 0
