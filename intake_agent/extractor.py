"""Extractors: the Claude-backed one and a deterministic rule-based mock.

Both return an ExtractionResult with a raw value and a self-reported confidence
for each field. Validation and gating happen downstream in the pipeline, so the
two extractors are interchangeable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

import anthropic

from .config import Settings, cost_usd
from .pdftext import collapse_fragments, extract_raw_text, has_usable_text, is_fragmented
from .schema import FIELDS, ExtractionResult, ModelExtraction, RawField

SYSTEM_PROMPT = """\
You extract structured fields from healthcare referral and prior-authorization documents
for an intake team. Return every field in the schema. For each field give the value and
your confidence (0.0 to 1.0) that the value is exactly right.

Rules:
- Only report what is written in the document. Never infer or invent a value. If a field
  is blank, illegible, or absent, return null (or an empty list for codes) with confidence 1.0
  if you are sure it is absent, lower if you are unsure.
- document_type: "referral" or "prior_auth" (precertification, prior authorization, letter of
  medical necessity, and utilization management requests are prior_auth).
- patient_name and referring_provider: "First Last", no titles (Dr.) or credentials (MD, DO, NP).
- Dates (patient_dob, request_date): ISO format YYYY-MM-DD. request_date is the date the
  request was written, signed, or submitted.
- member_id: as printed, uppercase, no spaces.
- referring_npi: the 10-digit NPI exactly as printed, even if it looks wrong.
- icd10_codes: every ICD-10-CM code, with the dot (e.g. M17.11). cpt_codes: every CPT/HCPCS code.
- urgency: "routine", "urgent", or "stat". Standard review = routine, expedited = urgent,
  emergent or STAT = stat.
- requested_service: a short description of the service as written.
- Lower your confidence when text is garbled, handwritten, cut off, or ambiguous.
"""

PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + json.dumps(ModelExtraction.model_json_schema(), sort_keys=True)).encode()
).hexdigest()[:16]


class Extractor(Protocol):
    model: str
    prompt_hash: str

    def extract(self, path: Path) -> ExtractionResult: ...


class ExtractionError(RuntimeError):
    pass


class ClaudeExtractor:
    """Extract with the Messages API and a pydantic structured-output schema."""

    prompt_hash = PROMPT_HASH

    def __init__(self, settings: Settings | None = None, client: anthropic.Anthropic | None = None):
        self.settings = settings or Settings()
        self.model = self.settings.model
        self.client = client or anthropic.Anthropic()

    def _content(self, path: Path) -> tuple[list[dict], bool]:
        text = collapse_fragments(extract_raw_text(path))
        if has_usable_text(text):
            return [
                {
                    "type": "text",
                    "text": f"<document_text>\n{text}\n</document_text>\n\n"
                    "Extract the fields from this document.",
                }
            ], False
        # No usable text layer (scan or image-only fax): send the PDF itself.
        data = base64.standard_b64encode(path.read_bytes()).decode()
        return [
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": data},
            },
            {"type": "text", "text": "Extract the fields from this scanned document."},
        ], True

    def extract(self, path: Path) -> ExtractionResult:
        content, used_doc = self._content(path)
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_format=ModelExtraction,
                output_config={"effort": self.settings.effort},
            )
        except anthropic.AuthenticationError:
            raise  # a bad key fails the whole run, not one document
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            # The SDK has already retried 429/5xx/connection errors by this point.
            raise ExtractionError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
        if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
            raise ExtractionError(f"{path.name}: no structured output ({response.stop_reason})")
        parsed: ModelExtraction = response.parsed_output
        fields = {
            name: RawField(
                value=getattr(parsed, name).value, model_confidence=getattr(parsed, name).confidence
            )
            for name in FIELDS
        }
        usage = response.usage
        return ExtractionResult(
            fields=fields,
            model=self.model,
            prompt_hash=self.prompt_hash,
            used_document_block=used_doc,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost_usd(self.model, usage.input_tokens, usage.output_tokens),
        )


# ------------------------------------------------------------------ mock


ICD_IN_TEXT = re.compile(r"\b[A-TV-Z][0-9][0-9AB]\.[0-9A-TV-Z]{1,4}\b")
CPT_IN_TEXT = re.compile(r"\b[0-9]{4}[0-9FTU]\b")
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
LONG_DATE = rf"{_MONTH} \d{{1,2}}, \d{{4}}"

# Field -> label regexes used on form-like documents (label on its own line or before ':').
LABELS: dict[str, list[str]] = {
    "patient_name": [r"patient name", r"member name", r"pt name", r"patient"],
    "patient_dob": [r"dob", r"date of birth"],
    "member_id": [r"member id", r"subscriber id", r"ins / id #"],
    "payer": [r"health plan", r"plan", r"payer"],
    "referring_provider": [r"referring provider", r"ordering physician", r"ref\. md", r"referring"],
    "referring_npi": [r"provider npi", r"ordering npi", r"npi #", r"npi"],
    "icd10_codes": [r"diagnosis \(icd-10\)", r"dx codes", r"dx", r"icd-10"],
    "cpt_codes": [r"procedure \(cpt\)", r"cpt/hcpcs", r"cpt"],
    "requested_service": [r"service requested", r"description of service", r"for", r"service"],
    "urgency": [r"urgency", r"review priority", r"priority"],
    "request_date": [r"date of request", r"submission date", r"date"],
}


def _is_label(value: str) -> bool:
    """True if value is itself a field label (blank field followed by the next label)."""
    v = value.strip().rstrip(":").lower()
    return any(re.fullmatch(p, v) for pats in LABELS.values() for p in pats) or v.endswith(":")


URGENCY_WORDS = [
    ("stat", "stat"),
    ("emergent", "stat"),
    ("expedited", "urgent"),
    ("urgent", "urgent"),
    ("standard", "routine"),
    ("routine", "routine"),
]


def _urgency(text: str) -> str | None:
    checked = re.search(r"\[X\]\s*(\w+)", text, re.I)
    if checked:
        text = checked.group(1)
    low = text.lower()
    for word, value in URGENCY_WORDS:
        if re.search(rf"\b{word}\b", low):
            return value
    return None


class MockExtractor:
    """Deterministic rule-based extractor. No network, no model.

    It reads labeled forms and the narrative letter layout with regexes, and reports
    a fixed confidence that depends on how it found the value: 0.95 for a labeled
    field, 0.80 for prose, 0.70 when the text layer was glyph-fragmented. It cannot
    read scans. It exists so the pipeline, review queue and eval run end to end
    without an API key, and as a baseline for the model.
    """

    model = "mock-rules-v1"
    prompt_hash = hashlib.sha256(b"mock-rules-v1").hexdigest()[:16]

    def extract(self, path: Path) -> ExtractionResult:
        raw = extract_raw_text(path)
        text = collapse_fragments(raw)
        if not has_usable_text(text):
            fields = {
                n: RawField(value=[] if n.endswith("codes") else None, model_confidence=0.0)
                for n in FIELDS
            }
        elif "To whom it may concern" in text:
            fields = self._from_prose(text)
        else:
            fields = self._from_form(text, 0.70 if is_fragmented(raw) else 0.95)
        return ExtractionResult(fields=fields, model=self.model, prompt_hash=self.prompt_hash)

    @staticmethod
    def _labeled(lines: list[str], patterns: list[str]) -> str | None:
        for pattern in patterns:
            rx = re.compile(rf"^{pattern}\s*(?::\s*(.*))?$", re.I)
            for i, line in enumerate(lines):
                m = rx.match(line.strip())
                if not m:
                    continue
                value = (m.group(1) or "").strip()
                if not value and i + 1 < len(lines):
                    value = lines[i + 1].strip()
                if _is_label(value) or set(value) <= set("_/ "):
                    return None  # the next line is another label, or the field is a blank rule
                return value
        return None

    def _from_form(self, text: str, conf: float) -> dict[str, RawField]:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        out: dict[str, RawField] = {}
        for name in FIELDS:
            if name == "document_type":
                low = text.lower()
                is_pa = any(
                    k in low
                    for k in ("[x] prior auth", "precertification", "prior authorization request")
                )
                out[name] = RawField(
                    value="prior_auth" if is_pa else "referral", model_confidence=conf
                )
                continue
            value = self._labeled(lines, LABELS[name])
            parsed: str | list[str] | None = value
            if name == "icd10_codes":
                parsed = ICD_IN_TEXT.findall(value or "")
            elif name == "cpt_codes":
                parsed = CPT_IN_TEXT.findall(value or "")
            elif name == "urgency":
                parsed = _urgency(value) if value else None
            elif name == "referring_npi" and value:
                parsed = value if re.fullmatch(r"\d{10}", value) else None
            empty = parsed is None or parsed == []
            out[name] = RawField(value=parsed, model_confidence=0.0 if empty else conf)
        return out

    def _from_prose(self, text: str) -> dict[str, RawField]:
        flat = re.sub(r"\s+", " ", text)
        conf = 0.80

        def grab(pattern: str) -> str | None:
            m = re.search(pattern, flat)
            return m.group(1).strip() if m else None

        lines = text.split("\n")
        provider = None
        for i, line in enumerate(lines):
            if line.strip() == "Sincerely," and i + 1 < len(lines):
                provider = lines[i + 1].strip()
        service = grab(r"I am requesting (.+?)(?: at the next| on an urgent| STAT,|\. )")
        cpt = grab(r"CPT ([\d, ]+)")
        values: dict[str, str | list[str] | None] = {
            "document_type": "prior_auth" if "prior authorization" in flat.lower() else "referral",
            "patient_name": grab(r"my patient ([A-Z][a-z]+ [A-Z][a-z]+)"),
            "patient_dob": grab(rf"DOB ({LONG_DATE})"),
            "member_id": grab(r"member ID ([A-Z0-9-]+)"),
            "payer": grab(r"insured with ([^.]+)\."),
            "referring_provider": provider,
            "referring_npi": grab(r"NPI (\d{10})"),
            "icd10_codes": ICD_IN_TEXT.findall(flat),
            "cpt_codes": CPT_IN_TEXT.findall(cpt or ""),
            "requested_service": service,
            "urgency": _urgency(grab(r"(I am requesting .+?\. )") or ""),
            "request_date": grab(rf"^.*?({LONG_DATE})"),
        }
        return {
            n: RawField(value=v, model_confidence=0.0 if v in (None, []) else conf)
            for n, v in values.items()
        }


def make_extractor(mock: bool, settings: Settings | None = None) -> Extractor:
    return MockExtractor() if mock else ClaudeExtractor(settings)
