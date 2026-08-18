"""Hybrid retrieval via Reciprocal Rank Fusion.

BM25 and dense retrieval fail on opposite inputs - one is blind to synonyms,
the other to rare exact strings - so combining them should beat either alone.
The question is *how* to combine them.

**Why RRF instead of blending scores.** BM25 scores are unbounded sums whose
scale depends on corpus statistics; cosine similarities live in [-1, 1] and
cluster tightly near the top. They are not comparable, and normalising them
(min-max, z-score) only makes them look comparable: the transform depends on
the score distribution of the particular query, so the same document can win or
lose depending on what else was retrieved alongside it.

RRF throws the scores away and fuses *ranks*:

    score(d) = sum over retrievers of 1 / (k + rank(d))

Rank is the only signal both retrievers express on the same scale. The constant
``k`` (60 by convention, from Cormack et al. 2009) damps the top ranks so that
a single retriever's first place cannot by itself outvote broad agreement
further down.
"""

from __future__ import annotations

from collections import defaultdict

from ..corpus import Chunk
from .base import Retriever, ScoredChunk


class HybridRetriever(Retriever):
    """Fuses any number of retrievers by rank.

    Args:
        retrievers: The retrievers to fuse.
        k: RRF damping constant. Lower makes top ranks more decisive.
        weights: Optional per-retriever multipliers, in the same order.
        candidates: How deep to read from each retriever before fusing.
            Must exceed the final ``k`` or fusion has nothing to disagree about.
    """

    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        k: int = 60,
        weights: list[float] | None = None,
        candidates: int = 50,
    ) -> None:
        if not retrievers:
            raise ValueError("hybrid retrieval needs at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must match the number of retrievers")

        self.retrievers = retrievers
        self.k = k
        self.weights = weights or [1.0] * len(retrievers)
        self.candidates = candidates
        parts = "+".join(r.name.split("(")[0] for r in retrievers)
        self.name = f"hybrid({parts},k={k})"

    def index(self, chunks: list[Chunk]) -> None:
        for retriever in self.retrievers:
            retriever.index(chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        fused: dict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}

        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            for rank, scored in enumerate(retriever.search(query, self.candidates), start=1):
                fused[scored.chunk.chunk_id] += weight / (self.k + rank)
                chunks[scored.chunk.chunk_id] = scored.chunk

        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return [ScoredChunk(chunk=chunks[cid], score=score) for cid, score in ranked[:k]]
