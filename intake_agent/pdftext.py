"""Text-layer extraction with pypdf."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

# Below this many word characters, treat the text layer as absent (scans, image-only faxes).
MIN_TEXT_CHARS = 40


def extract_raw_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def collapse_fragments(text: str) -> str:
    """Join runs of one-character lines.

    Glyph-by-glyph text (handwriting fonts, some fax software) comes out of pypdf as
    one character per line. Rejoining the run recovers the word.
    """
    out: list[str] = []
    run: list[str] = []
    for line in text.split("\n"):
        if len(line) == 1:
            run.append(line)
            continue
        if run:
            out.append("".join(run).strip())
            run = []
        out.append(line)
    if run:
        out.append("".join(run).strip())
    return "\n".join(out)


def is_fragmented(raw_text: str) -> bool:
    """True if most lines are single characters (a sign of a low-quality text layer)."""
    lines = [ln for ln in raw_text.split("\n") if ln]
    return bool(lines) and sum(len(ln) == 1 for ln in lines) / len(lines) > 0.5


def has_usable_text(text: str) -> bool:
    return len(re.findall(r"\w", text)) >= MIN_TEXT_CHARS
