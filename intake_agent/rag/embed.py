"""Embeddings.

HashingEmbedder is a feature-hashing bag of words and bigrams. It needs no model, no
network and no key, and it gives the same vector on every machine, which is what the
tests and CI need. It is lexical, not semantic: "knee replacement" will not match
"arthroplasty". A real embedding model plugs in through the Embedder protocol; the
pgvector column is sized from Embedder.dim.
"""

from __future__ import annotations

import hashlib
import itertools
import math
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
