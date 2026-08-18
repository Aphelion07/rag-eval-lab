"""The chunker contract.

Chunking is the least glamorous and most consequential decision in a retrieval
system. Split too finely and a chunk loses the context that made it meaningful;
too coarsely and the embedding averages several topics into a vector that
matches none of them well. This package exists so the choice can be measured
instead of argued about.

Sizes are in **characters**, not tokens. A token-accurate splitter would mean
shipping a tokenizer per model family, and the comparison here is between
splitting *strategies* - a bias that applies equally to every strategy does not
change their ranking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..corpus import Chunk, Document


class Chunker(ABC):
    """Splits a document into retrievable units."""

    name: str

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split raw text. Implementations only deal with strings."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document and attach identity to each piece."""
        return [
            Chunk(
                chunk_id=f"{document.doc_id}#{position}",
                doc_id=document.doc_id,
                text=piece,
                position=position,
            )
            for position, piece in enumerate(self.split(document.full_text))
            if piece.strip()
        ]

    def chunk_all(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for document in documents for chunk in self.chunk(document)]

    def __repr__(self) -> str:
        return self.name


def merge_with_overlap(units: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Greedily pack ``units`` into chunks of at most ``max_chars``.

    Overlap is applied by replaying whole trailing units from the previous
    chunk rather than by slicing mid-word. Overlap exists to stop an answer
    that straddles a boundary from being lost by both neighbours; cutting a
    sentence in half to achieve it defeats the purpose.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in units:
        unit_length = len(unit) + 1  # the joining space
        if current and length + unit_length > max_chars:
            chunks.append(" ".join(current))
            if overlap_chars > 0:
                carried: list[str] = []
                carried_length = 0
                for previous in reversed(current):
                    if carried_length + len(previous) + 1 > overlap_chars:
                        break
                    carried.insert(0, previous)
                    carried_length += len(previous) + 1
                current = carried
                length = carried_length
            else:
                current = []
                length = 0
        current.append(unit)
        length += unit_length

    if current:
        chunks.append(" ".join(current))
    return chunks
