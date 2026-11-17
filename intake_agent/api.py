"""FastAPI service over the intake pipeline.

    POST /extract   run the LangGraph pipeline on an uploaded PDF or text body
    POST /ask       cited answer over the indexed documents and payer policies
    GET  /review    documents waiting for a human
    GET  /healthz   liveness and kill switch state

Mock mode (rule-based extractor, extractive answerer) is on when INTAKE_MOCK is set
or ANTHROPIC_API_KEY is not. No API key is needed to run it.

The kill switch stops new work: /extract and /ask return 503 while it is engaged.
/review and /healthz keep answering so the queue can drain and operators can see why.
"""

import hashlib
import io
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import killswitch
from .config import Settings
from .extractor import ExtractionError, make_extractor
from .graph import KillSwitchEngaged, run_document
from .pipeline import open_state
from .rag import HashingEmbedder, RagIndex, load_corpus, make_answerer, make_store
from .schema import FIELDS

_TRUTHY = {"1", "true", "yes", "on"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    doc_id: str | None = None
    policy_only: bool = False
    k: int = Field(default=5, ge=1, le=20)


def _text_to_pdf(text: str) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 750
    for line in text.splitlines():
        if y < 50:
            c.showPage()
            y = 750
        c.drawString(40, y, line[:110])
        y -= 14
    c.save()
    return buf.getvalue()


def create_app(
    settings_factory: Callable[[], Settings] = Settings, mock: bool | None = None
) -> FastAPI:
    app = FastAPI(title="intake-agent", version="0.1.0")
    index_cache: dict[str, RagIndex] = {}

    def use_mock() -> bool:
        if mock is not None:
            return mock
        if os.environ.get("INTAKE_MOCK", "").strip().lower() in _TRUTHY:
            return True
        return not os.environ.get("ANTHROPIC_API_KEY")

    def get_settings() -> Settings:
        return settings_factory()

    Cfg = Annotated[Settings, Depends(get_settings)]

    def require_running(settings: Settings) -> None:
        reason = killswitch.engaged(settings)
        if reason:
            raise HTTPException(503, detail=f"kill switch engaged: {reason}")

    def get_index(settings: Settings) -> RagIndex:
        key = settings.pg_dsn or "memory"
        if key not in index_cache:
            emb = HashingEmbedder()
            idx = RagIndex(emb, make_store(emb.dim, settings.pg_dsn))
            if idx.store.count() == 0:
                data = Path(os.environ.get("INTAKE_DATA_DIR", "data/synthetic"))
                policies = Path(os.environ.get("INTAKE_POLICY_DIR", "data/policies"))
                sources, _ = load_corpus(data, policies)
                idx.build(sources)
            index_cache[key] = idx
        return index_cache[key]

    @app.get("/healthz")
    def healthz(settings: Cfg) -> dict:
        return {
            "status": "ok",
            "mode": "mock" if use_mock() else "model",
            "kill_switch": killswitch.engaged(settings) or "off",
        }

    @app.post("/extract")
    async def extract(request: Request, settings: Cfg, doc_id: str | None = None) -> dict:
        """Body is a PDF (Content-Type application/pdf) or plain text (text/plain)."""
        require_running(settings)
        body = await request.body()
        if not body:
            raise HTTPException(400, detail="empty body")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="upload too large")
        if body[:5] == b"%PDF-":
            pdf = body
        elif "pdf" in request.headers.get("content-type", ""):
            raise HTTPException(400, detail="body is not a valid PDF")
        else:
            pdf = _text_to_pdf(body.decode("utf-8", errors="replace"))
        doc_id = doc_id or "upload-" + hashlib.sha256(body).hexdigest()[:10]
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", doc_id):
            raise HTTPException(422, detail="doc_id must be 1-64 chars of letters, digits . _ -")
        _, queue = open_state(settings)
        if queue.has(doc_id):
            raise HTTPException(409, detail=f"document {doc_id} already processed")
        extractor = make_extractor(use_mock(), settings)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{doc_id}.pdf"
            path.write_bytes(pdf)
            try:
                state = run_document(path, extractor, queue, settings, doc_id)
            except KillSwitchEngaged as exc:
                raise HTTPException(503, detail=f"kill switch engaged: {exc}") from exc
            except ExtractionError as exc:
                raise HTTPException(502, detail=f"extraction failed: {exc}") from exc
        return {
            "doc_id": doc_id,
            "status": state["status"],
            "route": state["route"],
            "model": state["result"].model,
            "threshold": settings.threshold,
            "flagged_fields": state["flagged"],
            "fields": {
                name: {
                    "value": state["scored"][name].value,
                    "confidence": state["scored"][name].confidence,
                    "issues": state["scored"][name].issues,
                    "citation": state["citations"][name],
                }
                for name in FIELDS
            },
        }

    @app.post("/ask")
    def ask(req: AskRequest, settings: Cfg) -> dict:
        require_running(settings)
        m = use_mock()
        if not m and not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(503, detail="ANTHROPIC_API_KEY is not set")
        idx = get_index(settings)
        hits = idx.retrieve(
            req.question,
            k=req.k,
            doc_id=req.doc_id,
            prefix="policy:" if req.policy_only else None,
        )
        ans = make_answerer(m, settings).answer(req.question, hits)
        audit, _ = open_state(settings)
        audit.append(
            actor="api",
            action="rag_answer",
            doc_id=req.doc_id,
            model=ans.model,
            detail={
                "question": req.question,
                "grounded": ans.grounded,
                "citations": [c.model_dump() for c in ans.citations],
                "dropped_citations": ans.dropped_citations,
            },
        )
        return ans.model_dump()

    @app.get("/review")
    def review(settings: Cfg) -> dict:
        _, queue = open_state(settings)
        rows = queue.pending()
        return {"count": len(rows), "items": rows}

    return app


app = create_app()
