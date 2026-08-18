"""The embedder contract, plus a deterministic fake.

``HashingEmbedder`` is not a toy: it makes the entire test suite and CI run
without a model download, and it is a real (if weak) embedding method - a
hashing vectoriser over character n-grams. Because it is lexical, a dense
retriever built on it behaves measurably worse than one on real embeddings,
which is exactly what a test asserting "real embeddings beat hashing" needs.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    name: str
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Order of the output matches the input."""


class HashingEmbedder(Embedder):
    """Character n-gram hashing into a fixed-width vector.

    No training, no download, fully deterministic. Captures surface similarity
    ("password" vs "passwords") but nothing semantic - it has no way to relate
    "2FA" to "two-factor authentication".
    """

    def __init__(self, dimensions: int = 256, ngram: int = 3) -> None:
        self.dimensions = dimensions
        self.ngram = ngram
        self.name = f"hashing(dim={dimensions},n={ngram})"

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        cleaned = " ".join(text.lower().split())
        if not cleaned:
            return vector.tolist()

        for i in range(max(1, len(cleaned) - self.ngram + 1)):
            gram = cleaned[i : i + self.ngram]
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            # Sign from an independent byte, so colliding n-grams cancel as
            # often as they reinforce instead of inflating every bucket.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = float(np.linalg.norm(vector))
        return (vector / norm).tolist() if norm else vector.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
