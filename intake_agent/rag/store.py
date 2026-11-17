"""Vector stores: PostgreSQL + pgvector, and an in-memory fallback with the same interface.

The in-memory store is what the tests and the default CLI use, so nothing needs a
database to run. Set INTAKE_PG_DSN (docker compose does) to use pgvector instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from .corpus import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, vector: list[float], k: int = 5, doc_id: str | None = None, prefix: str | None = None
    ) -> list[Hit]:
        """Top k chunks by cosine similarity, optionally limited to a doc_id or doc_id prefix."""
        ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


class InMemoryStore:
    """Brute-force cosine similarity. Vectors are already unit length, so it's a dot product."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._rows[chunk.chunk_id] = (chunk, vec)

    def search(
        self, vector: list[float], k: int = 5, doc_id: str | None = None, prefix: str | None = None
    ) -> list[Hit]:
        hits = [
            Hit(chunk, sum(a * b for a, b in zip(vector, vec, strict=True)))
            for chunk, vec in self._rows.values()
            if (doc_id is None or chunk.doc_id == doc_id)
            and (prefix is None or chunk.doc_id.startswith(prefix))
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return hits[:k]

    def count(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()


def _vec_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


class PgVectorStore:
    """Chunks and embeddings in one PostgreSQL table, cosine search with an HNSW index."""

    def __init__(self, dsn: str, dim: int, table: str = "rag_chunks"):
        import psycopg  # optional dependency: pip install -e ".[pg]"

        if not table.isidentifier():
            raise ValueError(f"bad table name {table!r}")
        self.table = table
        self.dim = dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {table} (
                    chunk_id  text PRIMARY KEY,
                    doc_id    text NOT NULL,
                    title     text NOT NULL,
                    start_pos integer NOT NULL,
                    end_pos   integer NOT NULL,
                    body      text NOT NULL,
                    embedding vector({dim}) NOT NULL
                )"""
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_doc_idx ON {table} (doc_id)")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_hnsw_idx ON {table} "
                "USING hnsw (embedding vector_cosine_ops)"
            )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = [
            (c.chunk_id, c.doc_id, c.title, c.start, c.end, c.text, _vec_literal(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table}
                    (chunk_id, doc_id, title, start_pos, end_pos, body, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                      doc_id = EXCLUDED.doc_id, title = EXCLUDED.title,
                      start_pos = EXCLUDED.start_pos, end_pos = EXCLUDED.end_pos,
                      body = EXCLUDED.body, embedding = EXCLUDED.embedding""",
                rows,
            )

    def search(
        self, vector: list[float], k: int = 5, doc_id: str | None = None, prefix: str | None = None
    ) -> list[Hit]:
        lit = _vec_literal(vector)
        conds, args = [], []
        if doc_id:
            conds.append("doc_id = %s")
            args.append(doc_id)
        if prefix:
            conds.append("starts_with(doc_id, %s)")
            args.append(prefix)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        params: list = [lit, *args, lit, k]
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT chunk_id, doc_id, title, start_pos, end_pos, body,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {self.table} {where}
                    ORDER BY embedding <=> %s::vector, chunk_id
                    LIMIT %s""",
                params,
            )
            return [Hit(Chunk(*row[:6]), float(row[6])) for row in cur.fetchall()]

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])

    def clear(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"TRUNCATE {self.table}")


def make_store(dim: int, dsn: str | None = None, table: str = "rag_chunks") -> VectorStore:
    dsn = dsn if dsn is not None else os.environ.get("INTAKE_PG_DSN", "")
    return PgVectorStore(dsn, dim, table=table) if dsn else InMemoryStore()
