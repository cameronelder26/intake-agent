"""FastAPI service in mock mode: no API key, no network."""

import pytest
from fastapi.testclient import TestClient

from intake_agent.api import create_app


@pytest.fixture
def client(settings):
    return TestClient(create_app(lambda: settings, mock=True))


def _pdf(dataset, n=0) -> bytes:
    return sorted(dataset.glob("*.pdf"))[n].read_bytes()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "mode": "mock", "kill_switch": "off"}


def test_extract_pdf_returns_fields_citations_confidence(client, dataset):
    r = client.post(
        "/extract?doc_id=t1", content=_pdf(dataset), headers={"content-type": "application/pdf"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] in {"auto_accept", "human_review"}
    name = body["fields"]["patient_name"]
    assert name["confidence"] > 0 and name["citation"]["quote"]
    assert set(body["fields"]) >= {"cpt_codes", "referring_npi", "urgency"}
    again = client.post(
        "/extract?doc_id=t1", content=_pdf(dataset), headers={"content-type": "application/pdf"}
    )
    assert again.status_code == 409


def test_extract_plain_text(client):
    text = (
        "OUTPATIENT REFERRAL FORM\nPatient Name:\nJordan Stubbins\nDOB:\n06/23/1958\n"
        "Member ID:\nNWH633021001\nHealth Plan:\nNorthwind Health Plan\n"
        "Referring Provider:\nLogan Synthwood, DO\nProvider NPI:\n2975432313\n"
        "Diagnosis (ICD-10):\nM51.16\nProcedure (CPT):\n64483\n"
        "Service Requested:\nEpidural steroid injection\n"
        "Urgency:\n[X] Routine   [ ] Urgent   [ ] STAT\nDate of Request:\n06/07/2026\n"
    )
    r = client.post("/extract", content=text, headers={"content-type": "text/plain"})
    assert r.status_code == 200, r.text
    assert r.json()["fields"]["member_id"]["value"] == "NWH633021001"


def test_extract_rejects_bad_input(client):
    assert client.post("/extract", content=b"").status_code == 400
    r = client.post("/extract", content=b"nope", headers={"content-type": "application/pdf"})
    assert r.status_code == 400


def test_review_lists_queued_documents(client, dataset):
    for pdf in sorted(dataset.glob("*.pdf")):
        client.post(f"/extract?doc_id={pdf.stem}", content=pdf.read_bytes())
    body = client.get("/review").json()
    assert body["count"] == len(body["items"]) > 0
    assert {"doc_id", "open_fields", "min_confidence"} <= set(body["items"][0])


def test_ask_mock_returns_checked_citations(client):
    r = client.post(
        "/ask",
        json={
            "question": "Does Contoso Care require prior authorization for CPT 95806?",
            "policy_only": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grounded"] and body["citations"]
    assert all(c["doc_id"].startswith("policy:") and c["quote"] for c in body["citations"])


def test_kill_switch_returns_503(client, dataset, monkeypatch):
    monkeypatch.setenv("INTAKE_KILL_SWITCH", "1")
    assert client.post("/extract", content=_pdf(dataset)).status_code == 503
    assert client.post("/ask", json={"question": "anything at all"}).status_code == 503
    h = client.get("/healthz")
    assert h.status_code == 200 and h.json()["kill_switch"] != "off"
    assert client.get("/review").status_code == 200
