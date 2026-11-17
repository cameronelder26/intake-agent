"""Retrieval over intake documents and payer policies, with checked source citations.

corpus.py   load documents, split them into chunks that keep character offsets
embed.py    Embedder protocol, FastEmbedder (local bge-small), HashingEmbedder fallback
store.py    VectorStore protocol, InMemoryStore, PgVectorStore (PostgreSQL + pgvector)
answer.py   retrieve, answer, and verify every citation against the source text
"""

from .answer import NOT_FOUND, Answer, Citation, ClaudeAnswerer, MockAnswerer, make_answerer
from .corpus import Chunk, Source, chunk_source, load_corpus
from .embed import Embedder, FastEmbedder, HashingEmbedder, make_embedder, store_table
from .index import RagIndex
from .store import Hit, InMemoryStore, PgVectorStore, VectorStore, make_store

__all__ = [
    "NOT_FOUND",
    "Answer",
    "Chunk",
    "Citation",
    "ClaudeAnswerer",
    "Embedder",
    "FastEmbedder",
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
    "make_embedder",
    "make_store",
    "store_table",
]
