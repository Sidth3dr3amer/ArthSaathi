"""
Embeddings for semantic memory recall.

Pluggable by design, because none of the four configured LLM providers offers an
embeddings API (Groq and Cerebras have none; LLM7 is chat-only) and PyTorch is
not installed, so local sentence-transformers is unavailable without a ~2 GB
dependency.

Two backends:

  hashed  (default)  Deterministic hashed bag-of-words. Offline, zero-dependency,
                     stable across processes. Captures lexical overlap only --
                     "emergency fund" and "rainy day savings" will NOT match.
                     Good enough for recall-by-topic, tests, and the demo.

  openai  (opt-in)   Real dense embeddings via text-embedding-3-small. Gives true
                     semantic recall. Enable by setting OPENAI_API_KEY and
                     EMBEDDING_BACKEND=openai in .env -- no code change.

Both are normalised to unit length, so cosine distance in pgvector behaves
consistently whichever backend is active.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, Sequence

from ..common import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: pgvector columns are fixed-width, so switching backend requires a migration.
#: The hashed backend is sized to match, keeping both interchangeable at 384.
HASHED_DIM = 384
OPENAI_MODEL = "text-embedding-3-small"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def hashed_embedding(text: str, dim: int = HASHED_DIM) -> list[float]:
    """
    Deterministic hashed bag-of-words with sub-word shingles.

    Shingles let near-miss spellings ("emergency"/"emergencies") share buckets,
    which plain word hashing would not.
    """
    vector = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vector

    for token in tokens:
        pieces = [token]
        if len(token) > 4:
            pieces += [token[i:i + 4] for i in range(len(token) - 3)]
        for piece in pieces:
            digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

    return _normalise(vector)


def openai_embedding(text: str, dim: int = HASHED_DIM) -> list[float]:
    """Real semantic embedding. Requires OPENAI_API_KEY."""
    from openai import OpenAI

    api_key = config.env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "EMBEDDING_BACKEND=openai but OPENAI_API_KEY is not set. "
            "Uncomment it in .env or switch the backend back to 'hashed'."
        )
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=OPENAI_MODEL, input=text or " ", dimensions=dim
    )
    return _normalise(list(response.data[0].embedding))


_BACKENDS: dict[str, Callable[[str, int], list[float]]] = {
    "hashed": hashed_embedding,
    "openai": openai_embedding,
}


def active_backend() -> str:
    name = (config.env("EMBEDDING_BACKEND", "hashed") or "hashed").lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"unknown EMBEDDING_BACKEND {name!r}; expected one of {sorted(_BACKENDS)}"
        )
    return name


def embed(text: str, dim: int | None = None) -> list[float]:
    """Embed one string using the configured backend."""
    dim = dim or config.EMBEDDING_DIM
    return _BACKENDS[active_backend()](text, dim)


def embed_many(texts: Sequence[str], dim: int | None = None) -> list[list[float]]:
    return [embed(t, dim) for t in texts]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Similarity of two vectors. Both backends emit unit vectors, so this is a dot product."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    return max(-1.0, min(1.0, dot))
