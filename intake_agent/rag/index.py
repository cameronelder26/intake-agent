"""Build the index from sources and retrieve chunks for a question."""

from __future__ import annotations

from .corpus import Source, chunk_source
from .embed import Embedder
from .store import Hit, VectorStore


class RagIndex:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def build(self, sources: list[Source], max_chars: int = 400) -> int:
        chunks = [c for s in sources for c in chunk_source(s, max_chars)]
        self.store.add(chunks, [self.embedder.embed(c.embed_text) for c in chunks])
        return len(chunks)

    def retrieve(
        self, question: str, k: int = 5, doc_id: str | None = None, prefix: str | None = None
    ) -> list[Hit]:
        return self.store.search(self.embedder.embed(question), k=k, doc_id=doc_id, prefix=prefix)
