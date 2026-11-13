"""Deterministic field validators and normalizers.

The model reports its own confidence, but self-reports are not calibrated. These
checks are cheap and catch a class of errors the model cannot talk its way past:
a bad NPI checksum, a malformed code, a date of birth in the future.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from .schema import ENUMS, FIELDS, FieldValue

ICD10_RE = re.compile(r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$")
CPT_RE = re.compile(r"^[0-9]{4}[0-9FTU]$")
MEMBER_ID_RE = re.compile(r"^[A-Z0-9-]{6,20}$")

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m/%d/%y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


def luhn_ok(digits: str) -> bool:
    """Standard Luhn mod-10 check over a digit string."""
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def npi_check_digit(first_nine: str) -> str:
    """Check digit for an NPI. NPIs use Luhn over the 80840 prefix plus 9 digits."""
    for d in "0123456789":
        if luhn_ok("80840" + first_nine + d):
            return d
    raise ValueError(f"no check digit for {first_nine!r}")


def npi_is_valid(npi: str) -> bool:
    return len(npi) == 10 and npi[0] in "12" and luhn_ok("80840" + npi)


def parse_date(text: str) -> date | None:
    text = re.sub(r"\s+", " ", text.strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def normalize_date(text: str | None) -> str | None:
    if not text:
        return None
    parsed = parse_date(text)
    return parsed.isoformat() if parsed else text.strip()


def normalize_name(text: str | None) -> str | None:
    """'DOE, JANE A' -> 'Jane A Doe'. Titles and credentials are dropped."""
    if not text:
        return None
    text = re.sub(r"\b(Dr|Mr|Mrs|Ms)\.?\s+", "", text.strip(), flags=re.I)
    text = re.sub(r",?\s+(MD|DO|NP|PA-C|PA|FNP|DPM)\b\.?", "", text, flags=re.I)
    if "," in text:
        last, _, first = text.partition(",")
        text = f"{first.strip()} {last.strip()}"
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(
        part.capitalize() if part.isupper() or part.islower() else part for part in text.split(" ")
    )


def normalize_codes(codes: list[str] | None) -> list[str]:
    if not codes:
        return []
    out = {c.strip().upper().replace(" ", "") for c in codes if c and c.strip()}
    return sorted(out)


def normalize_value(name: str, value: FieldValue) -> FieldValue:
    """Canonical form used both by the pipeline and by the eval comparison."""
    kind = FIELDS[name]
    if kind.endswith("_list"):
        return normalize_codes(value if isinstance(value, list) else None)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    assert isinstance(value, str)
    if kind in ("date", "past_date"):
        return normalize_date(value)
    if name in ("patient_name", "referring_provider"):
        return normalize_name(value)
    if kind == "npi":
        return re.sub(r"\D", "", value) or None
    if kind == "enum":
        return value.strip().lower().replace("-", "_").replace(" ", "_")
    if name == "member_id":
        return value.strip().upper().replace(" ", "")
    return re.sub(r"\s+", " ", value).strip()


def validate(name: str, value: FieldValue, today: date | None = None) -> list[str]:
    """Return a list of human-readable issues. Empty list means the value passed."""
    today = today or date.today()
    kind = FIELDS[name]
    if kind.endswith("_list"):
        codes = value if isinstance(value, list) else []
        if not codes:
            return ["missing"]
        pattern = ICD10_RE if kind == "icd10_list" else CPT_RE
        bad = [c for c in codes if not pattern.match(c)]
        return [f"bad {kind.split('_')[0].upper()} format: {', '.join(bad)}"] if bad else []

    if value is None or value == "":
        return ["missing"]
    assert isinstance(value, str)

    if kind in ("date", "past_date"):
        parsed = parse_date(value)
        if parsed is None:
            return [f"unparseable date: {value}"]
        if kind == "past_date" and not (date(1900, 1, 1) <= parsed < today):
            return ["date of birth not in the past"]
        if kind == "date" and not (date(2000, 1, 1) <= parsed <= today):
            return ["request date outside 2000..today"]
        return []
    if kind == "npi":
        return [] if npi_is_valid(value) else ["NPI fails checksum"]
    if kind == "enum":
        return [] if value in ENUMS[name] else [f"not one of {', '.join(ENUMS[name])}"]
    if name == "member_id" and not MEMBER_ID_RE.match(value):
        return ["member ID format"]
    return []
