"""Dense retrieval: cosine similarity over embedding vectors.

Strong where BM25 is blind - it connects "turn on two-factor auth" to a
document that only says "enable 2FA", because the two land near each other in
vector space without sharing a single token.

Its own failure mode is the mirror image: it is weak on rare exact strings.
An error code, a CVE identifier or a config key carries almost no semantic
signal, so a dense retriever will happily rank a topically-similar document
above the one containing the literal token being searched for.

Search is a brute-force matrix product. At corpus sizes an evaluation harness
handles (thousands of chunks) that is faster than any approximate index, and it
returns exact results - which matters here, because an ANN index would make the
benchmark measure the index's recall rather than the retriever's.
"""

from __future__ import annotations

import numpy as np

from ..corpus import Chunk
from ..embedding.base import Embedder
from .base import Retriever, ScoredChunk


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.name = f"dense({embedder.name})"
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        """Unit-normalise rows so cosine similarity becomes a dot product."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        vectors = self.embedder.embed([c.text for c in chunks])
        self._matrix = self._normalise(np.asarray(vectors, dtype=np.float32))

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before search()")

        query_vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if norm:
            query_vector = query_vector / norm

        scores = self._matrix @ query_vector
        # argpartition finds the top k without sorting the whole array; the
        # full sort then applies only to those k.
        count = min(k, len(scores))
        top = np.argpartition(-scores, count - 1)[:count] if count else np.array([], dtype=int)
        ordered = top[np.argsort(-scores[top])]

        return [
            ScoredChunk(chunk=self._chunks[int(i)], score=float(scores[int(i)])) for i in ordered
        ]
