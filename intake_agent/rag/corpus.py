"""Load the corpus and split it into chunks.

Every chunk keeps the character offsets of its text in the source document, so a
citation can point at an exact span (doc_id, start, end) and be checked later with
source.text[start:end].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..pdftext import collapse_fragments, extract_raw_text, has_usable_text


@dataclass(frozen=True)
class Source:
    doc_id: str
    title: str
    kind: str  # "intake" or "policy"
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    start: int
    end: int
    text: str

    @property
    def embed_text(self) -> str:
        # The title goes into the embedding (not the span) so a chunk from the
        # Northwind policy still says "Northwind" even when its lines don't.
        return f"{self.title}\n{self.text}"


def chunk_source(source: Source, max_chars: int = 400) -> list[Chunk]:
    """Line-aligned chunks of at most max_chars (a single longer line becomes its own chunk)."""
    chunks: list[Chunk] = []
    text = source.text
    start = None
    end = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        line_start, line_end = pos, pos + len(line.rstrip("\n"))
        pos += len(line)
        if not line.strip():
            continue
        if start is not None and line_end - start > max_chars:
            chunks.append(_chunk(source, len(chunks), start, end))
            start = None
        if start is None:
            start = line_start
        end = line_end
    if start is not None:
        chunks.append(_chunk(source, len(chunks), start, end))
    return chunks


def _chunk(source: Source, n: int, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{source.doc_id}#{n}",
        doc_id=source.doc_id,
        title=source.title,
        start=start,
        end=end,
        text=source.text[start:end],
    )


def load_corpus(
    data_dir: Path | None = Path("data/synthetic"),
    policy_dir: Path | None = Path("data/policies"),
) -> tuple[list[Source], list[str]]:
    """Return (sources, skipped doc ids). Scans with no text layer are skipped, not OCR'd."""
    sources: list[Source] = []
    skipped: list[str] = []
    if data_dir is not None:
        for pdf in sorted(data_dir.glob("*.pdf")):
            text = collapse_fragments(extract_raw_text(pdf))
            if not has_usable_text(text):
                skipped.append(pdf.stem)
                continue
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            sources.append(Source(pdf.stem, f"{pdf.stem} {first}", "intake", text))
    if policy_dir is not None:
        for md in sorted(policy_dir.glob("*.md")):
            text = md.read_text()
            first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
            sources.append(Source(f"policy:{md.stem}", first, "policy", text))
    return sources, skipped
