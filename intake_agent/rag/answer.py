"""Answer a question from retrieved chunks, with citations that are checked, not trusted.

Both answerers return the same Answer. A citation is kept only if its quote is found
verbatim in a chunk that was actually retrieved for this question; the span is then
computed from the chunk's offsets, so (doc_id, start, end) always points at real text.
If nothing survives that check the answer becomes NOT_FOUND instead of an uncited claim.
"""

from __future__ import annotations

from typing import Protocol

import anthropic
from pydantic import BaseModel

from ..config import Settings, cost_usd, estimate_tokens
from .embed import TOKEN, terms
from .store import Hit

NOT_FOUND = "Not found in the indexed documents."

SYSTEM_PROMPT = """\
You answer questions for a healthcare intake reviewer, using only the sources provided.

Rules:
- Use only the sources. If they don't contain the answer, set found to false, answer
  with an empty string, and return no citations. Never guess.
- Every answer needs at least one citation: the source's chunk_id and a quote copied
  exactly, character for character, from that source. Keep quotes short, one line or
  phrase.
- Answer in one or two plain sentences.
"""


class ModelCitation(BaseModel):
    chunk_id: str
    quote: str


class ModelAnswer(BaseModel):
    """Structured output schema for the answer call."""

    found: bool
    answer: str
    citations: list[ModelCitation]


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    start: int
    end: int
    quote: str


class Answer(BaseModel):
    question: str
    text: str
    grounded: bool
    citations: list[Citation]
    retrieved: list[str]
    dropped_citations: int = 0
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    cost_usd: float = 0.0


class Answerer(Protocol):
    model: str

    def answer(self, question: str, hits: list[Hit]) -> Answer: ...


def build_user_prompt(question: str, hits: list[Hit]) -> str:
    parts = ["<sources>"]
    for h in hits:
        c = h.chunk
        parts.append(f'<source chunk_id="{c.chunk_id}" title="{c.title}">\n{c.text}\n</source>')
    parts += ["</sources>", "", f"Question: {question}"]
    return "\n".join(parts)


def verify_citation(hits: list[Hit], chunk_id: str, quote: str) -> Citation | None:
    quote = quote.strip()
    if not quote:
        return None
    for h in hits:
        if h.chunk.chunk_id == chunk_id:
            i = h.chunk.text.find(quote)
            if i < 0:
                return None
            start = h.chunk.start + i
            return Citation(
                doc_id=h.chunk.doc_id,
                chunk_id=chunk_id,
                start=start,
                end=start + len(quote),
                quote=quote,
            )
    return None  # cited a chunk that wasn't retrieved


def finalize(
    question: str,
    hits: list[Hit],
    parsed: ModelAnswer,
    model: str,
    **usage,
) -> Answer:
    verified = [verify_citation(hits, c.chunk_id, c.quote) for c in parsed.citations]
    kept = [c for c in verified if c is not None]
    grounded = parsed.found and bool(kept) and bool(parsed.answer.strip())
    return Answer(
        question=question,
        text=parsed.answer.strip() if grounded else NOT_FOUND,
        grounded=grounded,
        citations=kept if grounded else [],
        retrieved=[h.chunk.chunk_id for h in hits],
        dropped_citations=len(verified) - len(kept),
        model=model,
        **usage,
    )


class ClaudeAnswerer:
    def __init__(self, settings: Settings | None = None, client: anthropic.Anthropic | None = None):
        self.settings = settings or Settings()
        self.model = self.settings.model
        self.client = client or anthropic.Anthropic()

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return finalize(
                question, hits, ModelAnswer(found=False, answer="", citations=[]), self.model
            )
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, hits)}],
            output_format=ModelAnswer,
            output_config={"effort": self.settings.effort},
        )
        parsed = response.parsed_output or ModelAnswer(found=False, answer="", citations=[])
        u = response.usage
        return finalize(
            question,
            hits,
            parsed,
            self.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cost_usd=cost_usd(self.model, u.input_tokens, u.output_tokens),
        )


def _is_label(line: str) -> bool:
    s = line.strip()
    return s.endswith(":") or (s.isupper() and len(s) < 40 and not any(ch.isdigit() for ch in s))


def _weight(term_set: set[str]) -> float:
    # Codes and IDs (anything with a digit) say much more about which line is meant.
    return sum(3.0 if any(ch.isdigit() for ch in t) else 1.0 for t in term_set)


class MockAnswerer:
    """Extractive answerer, no model. Picks the best-matching line and quotes it.

    Title words decide which document a chunk belongs to, and the remaining question
    words decide which line inside it answers the question. A "Label:" line is joined
    with the value on the next line. If the best line covers less than half of the
    non-title question words, it says NOT_FOUND. It is a baseline and a way to run the
    whole retrieval and citation path in tests and CI without an API key.
    """

    model = "mock-extractive-v1"

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        q = set(terms(question))
        best: tuple[float, int, int] | None = None
        pick: tuple[Hit, str, str] | None = None
        for rank, hit in enumerate(hits):
            title = set(terms(hit.chunk.title))
            want = q - title
            for pos, (quote, shown) in enumerate(self._units(hit.chunk.text)):
                unit = set(terms(shown))
                own = _weight(want & unit)
                if not own or own < 0.5 * _weight(want):
                    continue
                key = (_weight(q & (unit | title)), -rank, -pos)
                if best is None or key > best:
                    best, pick = key, (hit, quote, shown)
        if pick is None:
            parsed = ModelAnswer(found=False, answer="", citations=[])
        else:
            hit, quote, shown = pick
            parsed = ModelAnswer(
                found=True,
                answer=shown,
                citations=[ModelCitation(chunk_id=hit.chunk.chunk_id, quote=quote)],
            )
        prompt = SYSTEM_PROMPT + build_user_prompt(question, hits)
        return finalize(
            question,
            hits,
            parsed,
            self.model,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(parsed.model_dump_json()),
            tokens_estimated=True,
        )

    @staticmethod
    def _units(text: str) -> list[tuple[str, str]]:
        """(exact quote, display text) for each answerable line or label/value pair."""
        lines = [ln for ln in text.split("\n") if ln.strip()]
        units = []
        for i, line in enumerate(lines):
            if _is_label(line):
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if nxt and not _is_label(nxt) and TOKEN.search(nxt.lower()):
                    units.append((f"{line}\n{nxt}", f"{line.strip().rstrip(':')}: {nxt.strip()}"))
                continue  # a label with no value is a blank field, not an answer
            units.append((line, line.strip().lstrip("-# ").strip()))
        return units


def make_answerer(mock: bool, settings: Settings | None = None) -> Answerer:
    return MockAnswerer() if mock else ClaudeAnswerer(settings)
