"""Retrieval over intake documents and payer policies, with checked source citations.

corpus.py   load documents, split them into chunks that keep character offsets
embed.py    Embedder protocol and the deterministic HashingEmbedder
store.py    VectorStore protocol, InMemoryStore, PgVectorStore (PostgreSQL + pgvector)
answer.py   retrieve, answer, and verify every citation against the source text
"""

from .answer import NOT_FOUND, Answer, Citation, ClaudeAnswerer, MockAnswerer, make_answerer
from .corpus import Chunk, Source, chunk_source, load_corpus
from .embed import Embedder, HashingEmbedder
from .index import RagIndex
from .store import Hit, InMemoryStore, PgVectorStore, VectorStore, make_store

__all__ = [
    "NOT_FOUND",
    "Answer",
    "Chunk",
    "Citation",
    "ClaudeAnswerer",
    "Embedder",
    "HashingEmbedder",
    "Hit",
    "InMemoryStore",
    "MockAnswerer",
    "PgVectorStore",
    "RagIndex",
    "Source",
    "VectorStore",
    "chunk_source",
    "load_corpus",
    "make_answerer",
    "make_store",
]
