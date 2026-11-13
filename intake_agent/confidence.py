"""Combine the model's self-reported confidence with deterministic validators."""

from __future__ import annotations

from datetime import date

from .schema import RawField, ScoredField
from .validators import normalize_value, validate

# A value that fails a validator can never clear the review threshold,
# no matter how sure the model says it is.
VALIDATOR_FAIL_CAP = 0.30


def score_field(name: str, raw: RawField, today: date | None = None) -> ScoredField:
    value = normalize_value(name, raw.value)
    model_conf = min(max(raw.model_confidence, 0.0), 1.0)
    issues = validate(name, value, today)
    if "missing" in issues:
        confidence = 0.0
    elif issues:
        confidence = min(model_conf, VALIDATOR_FAIL_CAP)
    else:
        confidence = model_conf
    return ScoredField(
        name=name,
        value=value,
        model_confidence=model_conf,
        confidence=round(confidence, 4),
        issues=issues,
    )


def score_all(fields: dict[str, RawField], today: date | None = None) -> dict[str, ScoredField]:
    return {name: score_field(name, raw, today) for name, raw in fields.items()}


def needs_review(scored: dict[str, ScoredField], threshold: float) -> list[str]:
    """Names of fields below threshold. Empty list means the document can auto-approve."""
    return [name for name, f in scored.items() if f.confidence < threshold]
