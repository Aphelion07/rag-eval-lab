"""The retriever contract.

Two methods, because that is the whole shape of the problem: build an index,
then answer queries against it. Keeping them separate lets the harness charge
index time and query time to different columns - a retriever that is fast to
query but takes a minute to build is a different trade-off from one that is the
reverse, and a single "time" number would hide that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..corpus import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


class Retriever(ABC):
    name: str

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        """Build whatever structure this retriever searches."""

    @abstractmethod
    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return the top ``k`` chunks, best first."""

    def search_documents(self, query: str, k: int) -> list[str]:
        """Top ``k`` *document* ids, deduplicated, preserving rank order.

        Metrics are scored per document, not per chunk: retrieving three chunks
        of the same document is one useful result, not three. Over-fetching by
        4x before deduplication keeps a chunker that produces many small pieces
        from being unfairly truncated - without it, a fine-grained strategy
        could spend its whole budget on one document.
        """
        seen: list[str] = []
        for scored in self.search(query, k * 4):
            if scored.doc_id not in seen:
                seen.append(scored.doc_id)
            if len(seen) >= k:
                break
        return seen

    def __repr__(self) -> str:
        return self.name
