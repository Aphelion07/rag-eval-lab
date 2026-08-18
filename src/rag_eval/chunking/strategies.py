"""Concrete chunking strategies, ordered from naive to structure-aware."""

from __future__ import annotations

import re

from .base import Chunker, merge_with_overlap

# Sentence boundary: a full stop, question or exclamation mark followed by
# whitespace and a capital letter or digit. Deliberately not a full NLP
# sentence splitter - the failure cases (abbreviations, "e.g.") are shared by
# every strategy that uses it, so they do not skew the comparison, and avoiding
# an NLP dependency keeps the project installable in seconds.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


class FixedSizeChunker(Chunker):
    """Cuts every ``size`` characters, ignoring all structure.

    The baseline. Its weakness is the point of including it: it will happily
    end a chunk mid-sentence, and the benchmark shows what that costs.
    """

    def __init__(self, size: int = 512, overlap: int = 64) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed(size={size},overlap={overlap})"

    def split(self, text: str) -> list[str]:
        stride = self.size - self.overlap
        return [text[i : i + self.size] for i in range(0, len(text), stride)]


class SentenceChunker(Chunker):
    """Packs whole sentences up to a character budget.

    Never splits mid-sentence, so every chunk is a readable unit. Costs some
    size uniformity in exchange.
    """

    def __init__(self, size: int = 512, overlap: int = 64) -> None:
        self.size = size
        self.overlap = overlap
        self.name = f"sentence(size={size},overlap={overlap})"

    def split(self, text: str) -> list[str]:
        sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(text) if s.strip()]
        return merge_with_overlap(sentences, self.size, self.overlap)


class RecursiveChunker(Chunker):
    """Splits on the largest structural boundary that fits, descending as needed.

    Paragraphs first, then sentences, then words. This is what most RAG
    tutorials reach for; including it means the benchmark can say whether the
    extra machinery earns its keep on a given corpus, rather than assuming it.
    """

    def __init__(self, size: int = 512, overlap: int = 64) -> None:
        self.size = size
        self.overlap = overlap
        self.name = f"recursive(size={size},overlap={overlap})"

    def split(self, text: str) -> list[str]:
        return self._split_recursive(text, ["\n\n", "\n", ". ", " "])

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.size:
            return [text]
        if not separators:
            # No separator left: fall back to a hard cut rather than emitting an
            # oversized chunk that would silently blow past the size budget.
            stride = self.size - self.overlap
            return [text[i : i + self.size] for i in range(0, len(text), stride)]

        separator, *rest = separators
        parts = [p for p in text.split(separator) if p.strip()]
        if len(parts) == 1:
            return self._split_recursive(text, rest)

        pieces: list[str] = []
        for part in parts:
            pieces.extend([part] if len(part) <= self.size else self._split_recursive(part, rest))
        return merge_with_overlap(pieces, self.size, self.overlap)


class ParagraphChunker(Chunker):
    """One chunk per paragraph, with no size budget at all.

    Included as the honest control: on a corpus of short, well-structured
    documents it is often competitive with everything above, and a benchmark
    that omits the trivial option overstates the value of the clever ones.
    """

    name = "paragraph"

    def split(self, text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
