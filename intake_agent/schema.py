"""Field definitions and the structured-output models used for extraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FieldKind = Literal["text", "date", "past_date", "npi", "icd10_list", "cpt_list", "enum"]

# name -> kind. Order is the order fields are shown in the CLI and reports.
FIELDS: dict[str, FieldKind] = {
    "document_type": "enum",
    "patient_name": "text",
    "patient_dob": "past_date",
    "member_id": "text",
    "payer": "text",
    "referring_provider": "text",
    "referring_npi": "npi",
    "icd10_codes": "icd10_list",
    "cpt_codes": "cpt_list",
    "requested_service": "text",
    "urgency": "enum",
    "request_date": "date",
}

ENUMS: dict[str, tuple[str, ...]] = {
    "document_type": ("referral", "prior_auth"),
    "urgency": ("routine", "urgent", "stat"),
}

LIST_FIELDS = frozenset(name for name, kind in FIELDS.items() if kind.endswith("_list"))

FieldValue = str | list[str] | None


class StrField(BaseModel):
    """A scalar field as returned by the model."""

    value: str | None = Field(description="Value exactly as normalized per instructions, or null")
    confidence: float = Field(description="Your confidence from 0.0 to 1.0 that value is correct")


class ListField(BaseModel):
    """A list field (codes) as returned by the model."""

    value: list[str] = Field(description="All codes found, normalized; empty list if none")
    confidence: float = Field(description="Your confidence from 0.0 to 1.0 that the list is right")


class ModelExtraction(BaseModel):
    """Schema passed to the Messages API as the structured output format."""

    document_type: StrField
    patient_name: StrField
    patient_dob: StrField
    member_id: StrField
    payer: StrField
    referring_provider: StrField
    referring_npi: StrField
    icd10_codes: ListField
    cpt_codes: ListField
    requested_service: StrField
    urgency: StrField
    request_date: StrField


class RawField(BaseModel):
    """Extractor output for one field, before validation."""

    value: FieldValue
    model_confidence: float


class ScoredField(BaseModel):
    """One field after validators have been applied."""

    name: str
    value: FieldValue
    model_confidence: float
    confidence: float
    issues: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Everything the pipeline needs from one extractor call."""

    fields: dict[str, RawField]
    model: str
    prompt_hash: str
    used_document_block: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
