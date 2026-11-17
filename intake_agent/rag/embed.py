"""Embeddings.

HashingEmbedder is a feature-hashing bag of words and bigrams. It needs no model, no
network and no key, and it gives the same vector on every machine, which is what the
tests and CI need. It is lexical, not semantic: "knee replacement" will not match
"arthroplasty". A real embedding model plugs in through the Embedder protocol; the
pgvector column is sized from Embedder.dim.

FastEmbedder runs BAAI/bge-small-en-v1.5 locally through ONNX (no API key, no network
after the first model download). make_embedder() picks it when it is installed and the
model loads, and falls back to HashingEmbedder otherwise. INTAKE_EMBEDDER=hashing forces
the fallback (CI does this so it needs no download); INTAKE_EMBEDDER=fastembed makes a
missing model an error instead of a silent fallback.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
from typing import Protocol

TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?")

_STOP = """a an and are as at be by do does for from has have how i in is it its of on or
our that the this to was we what when where which who why will with"""
STOPWORDS = frozenset(_STOP.split())


def stem(token: str) -> str:
    """Light suffix stripping so 'required' meets 'require' and 'requests' meets 'request'."""
    if len(token) <= 4 or any(ch.isdigit() for ch in token):
        return token
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return token[:-1] if len(token) > 4 and token.endswith("e") else token


def terms(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-v1-{dim}"

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = terms(text)
        features = [(t, 1.0) for t in toks] + [
            (f"{a} {b}", 0.5) for a, b in itertools.pairwise(toks)
        ]
        for feat, weight in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += weight if (h >> 40) & 1 else -weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class FastEmbedder:
    """Local BAAI/bge-small-en-v1.5 (384 dims, unit-length vectors) via fastembed.

    Install with: pip install -e ".[embeddings]". The model (~130 MB) is downloaded once
    and cached.
    """

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model: str = MODEL):
        from fastembed import TextEmbedding  # optional dependency

        self._model = TextEmbedding(model)
        self.name = f"fastembed-{model.split('/')[-1]}"
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, text: str) -> list[float]:
        vec = next(iter(self._model.embed([text])))
        return [float(v) for v in vec]


def make_embedder(kind: str | None = None) -> Embedder:
    """INTAKE_EMBEDDER: auto (default), hashing, or fastembed."""
    kind = (kind or os.environ.get("INTAKE_EMBEDDER") or "auto").lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedder()
    if kind != "auto":
        raise ValueError(f"INTAKE_EMBEDDER must be auto, hashing or fastembed, got {kind!r}")
    try:
        return FastEmbedder()
    except Exception:  # not installed, or the model could not be loaded offline
        return HashingEmbedder()


def store_table(emb: Embedder) -> str:
    """pgvector table name per embedder, so vectors from two models never share a table."""
    if isinstance(emb, HashingEmbedder):
        return "rag_chunks"
    return "rag_chunks_" + re.sub(r"[^a-z0-9]+", "_", emb.name.lower()).strip("_")
