"""Chunking behaviour and the invariants every strategy must hold."""

from __future__ import annotations

import pytest

from rag_eval.chunking import (
    Chunker,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    SentenceChunker,
    merge_with_overlap,
)
from rag_eval.corpus import Document

PROSE = (
    "Authentication verifies identity. It answers who you are. "
    "Authorization is different. It decides what you may do.\n\n"
    "The two are routinely confused. That confusion causes real bugs. "
    "An API that checks a session but not ownership has authenticated only.\n\n"
    "Checks belong on the server. Hiding a button is not access control."
)

ALL_STRATEGIES: list[Chunker] = [
    FixedSizeChunker(size=120, overlap=20),
    SentenceChunker(size=120, overlap=20),
    RecursiveChunker(size=120, overlap=20),
    ParagraphChunker(),
]


@pytest.fixture
def document() -> Document:
    return Document(doc_id="auth", title="Authentication", text=PROSE)


class TestSharedInvariants:
    """Properties that must hold no matter which strategy is used, since the
    benchmark compares them against each other."""

    @pytest.mark.parametrize("chunker", ALL_STRATEGIES, ids=lambda c: c.name)
    def test_produces_chunks(self, chunker: Chunker, document: Document) -> None:
        assert chunker.chunk(document)

    @pytest.mark.parametrize("chunker", ALL_STRATEGIES, ids=lambda c: c.name)
    def test_every_chunk_knows_its_document(self, chunker: Chunker, document: Document) -> None:
        assert all(c.doc_id == "auth" for c in chunker.chunk(document))

    @pytest.mark.parametrize("chunker", ALL_STRATEGIES, ids=lambda c: c.name)
    def test_chunk_ids_are_unique(self, chunker: Chunker, document: Document) -> None:
        chunks = chunker.chunk(document)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    @pytest.mark.parametrize("chunker", ALL_STRATEGIES, ids=lambda c: c.name)
    def test_no_empty_chunks(self, chunker: Chunker, document: Document) -> None:
        assert all(c.text.strip() for c in chunker.chunk(document))

    @pytest.mark.parametrize("chunker", ALL_STRATEGIES, ids=lambda c: c.name)
    def test_title_is_included(self, chunker: Chunker, document: Document) -> None:
        """The title carries topical signal, so it must reach the index."""
        chunks = chunker.chunk(document)
        assert any("Authentication" in c.text for c in chunks)

    @pytest.mark.parametrize("chunker", ALL_STRATEGIES, ids=lambda c: c.name)
    def test_positions_are_sequential(self, chunker: Chunker, document: Document) -> None:
        positions = [c.position for c in chunker.chunk(document)]
        assert positions == sorted(positions)


class TestFixedSize:
    def test_respects_the_size_budget(self) -> None:
        chunks = FixedSizeChunker(size=50, overlap=0).split("x" * 200)
        assert all(len(c) <= 50 for c in chunks)

    def test_overlap_repeats_content(self) -> None:
        chunks = FixedSizeChunker(size=20, overlap=5).split("abcdefghij" * 5)
        assert chunks[0][-5:] == chunks[1][:5]

    def test_rejects_overlap_larger_than_size(self) -> None:
        with pytest.raises(ValueError, match="overlap must be smaller"):
            FixedSizeChunker(size=10, overlap=10)

    def test_splits_mid_sentence(self) -> None:
        """The documented weakness, asserted so the benchmark's premise holds:
        this strategy really does cut sentences in half."""
        text = "The quick brown fox jumps over the lazy dog and keeps running."
        chunks = FixedSizeChunker(size=20, overlap=0).split(text)
        assert not chunks[0].rstrip().endswith(".")


class TestSentence:
    def test_never_splits_mid_sentence(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        for chunk in SentenceChunker(size=45, overlap=0).split(text):
            assert chunk.rstrip().endswith(".")

    def test_packs_multiple_sentences_when_they_fit(self) -> None:
        text = "One. Two. Three."
        assert len(SentenceChunker(size=200, overlap=0).split(text)) == 1


class TestRecursive:
    def test_short_text_stays_whole(self) -> None:
        assert RecursiveChunker(size=500).split("Short text.") == ["Short text."]

    def test_prefers_paragraph_boundaries(self) -> None:
        text = "A" * 60 + "\n\n" + "B" * 60
        chunks = RecursiveChunker(size=80, overlap=0).split(text)
        assert len(chunks) == 2
        assert set(chunks[0]) == {"A"}

    def test_falls_back_to_hard_cut_without_separators(self) -> None:
        """A single unbroken token longer than the budget must still be split,
        not emitted oversized."""
        chunks = RecursiveChunker(size=30, overlap=0).split("x" * 200)
        assert all(len(c) <= 30 for c in chunks)


class TestParagraph:
    def test_one_chunk_per_paragraph(self) -> None:
        assert ParagraphChunker().split("First para.\n\nSecond para.\n\nThird.") == [
            "First para.",
            "Second para.",
            "Third.",
        ]

    def test_ignores_blank_runs(self) -> None:
        assert len(ParagraphChunker().split("A\n\n\n\n\nB")) == 2


class TestMergeWithOverlap:
    def test_packs_within_budget(self) -> None:
        merged = merge_with_overlap(["aaa", "bbb", "ccc"], 8, 0)
        assert all(len(m) <= 8 for m in merged)

    def test_overlap_replays_whole_units(self) -> None:
        """Overlap must not slice mid-word - a half word helps no retriever."""
        merged = merge_with_overlap(["alpha", "beta", "gamma", "delta"], 12, 6)
        assert all(all(word.isalpha() for word in chunk.split()) for chunk in merged)

    def test_rejects_overlap_at_or_above_max(self) -> None:
        with pytest.raises(ValueError, match="overlap_chars must be smaller"):
            merge_with_overlap(["a"], 10, 10)

    def test_rejects_non_positive_max(self) -> None:
        with pytest.raises(ValueError, match="max_chars must be positive"):
            merge_with_overlap(["a"], 0, 0)

    def test_empty_input(self) -> None:
        assert merge_with_overlap([], 10, 2) == []
