"""Retriever behaviour, including the failure modes that motivate the hybrid."""

from __future__ import annotations

import pytest

from rag_eval.corpus import Chunk
from rag_eval.embedding import HashingEmbedder
from rag_eval.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ScoredChunk,
    tokenize,
)


def chunk(chunk_id: str, text: str, doc_id: str | None = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id or chunk_id, text=text, position=0)


CORPUS = [
    chunk("ssh", "SSH provides encrypted remote shell access on TCP port 22."),
    chunk("tls", "TLS establishes an authenticated channel using certificates."),
    chunk("sql", "SQL injection concatenates untrusted input into a database query."),
    chunk("cmd", "Command injection passes untrusted input to a system shell."),
    chunk("bm25", "BM25 ranks documents by term frequency and inverse document frequency."),
]


class TestTokenize:
    def test_lowercases_and_splits(self) -> None:
        assert tokenize("SSH Port 22!", remove_stopwords=False) == ["ssh", "port", "22"]

    def test_removes_stopwords_by_default(self) -> None:
        assert "the" not in tokenize("the port")

    def test_can_keep_stopwords(self) -> None:
        assert "the" in tokenize("the port", remove_stopwords=False)

    def test_drops_punctuation(self) -> None:
        assert tokenize("a.b,c!") == ["b", "c"]


class TestBM25:
    @pytest.fixture
    def retriever(self) -> BM25Retriever:
        r = BM25Retriever()
        r.index(CORPUS)
        return r

    def test_ranks_the_exact_match_first(self, retriever: BM25Retriever) -> None:
        assert retriever.search("SSH port 22", 3)[0].chunk.chunk_id == "ssh"

    def test_returns_at_most_k(self, retriever: BM25Retriever) -> None:
        assert len(retriever.search("injection", 2)) == 2

    def test_scores_are_positive_and_descending(self, retriever: BM25Retriever) -> None:
        results = retriever.search("injection untrusted input", 5)
        scores = [r.score for r in results]
        assert all(s > 0 for s in scores)
        assert scores == sorted(scores, reverse=True)

    def test_unmatched_query_returns_nothing(self, retriever: BM25Retriever) -> None:
        assert retriever.search("quantum tunnelling photosynthesis", 5) == []

    def test_rare_terms_outweigh_common_ones(self, retriever: BM25Retriever) -> None:
        """IDF at work: 'injection' appears in two documents, '22' in one, so
        the rare term should be the decisive signal."""
        assert retriever.search("injection 22", 1)[0].chunk.chunk_id == "ssh"

    def test_searching_before_indexing_raises(self) -> None:
        with pytest.raises(RuntimeError, match="index\\(\\) must be called"):
            BM25Retriever().search("anything", 1)

    def test_is_blind_to_synonyms(self) -> None:
        """The documented weakness. This is *why* the dense retriever exists,
        so it is asserted rather than assumed."""
        retriever = BM25Retriever()
        retriever.index([chunk("doc", "Enable 2FA in your account settings.")])
        assert retriever.search("two-factor authentication", 5) == []

    def test_b_parameter_changes_length_normalisation(self) -> None:
        short = chunk("short", "injection")
        long = chunk("long", "injection " + "filler words here " * 30)

        no_penalty = BM25Retriever(b=0.0)
        no_penalty.index([short, long])
        full_penalty = BM25Retriever(b=1.0)
        full_penalty.index([short, long])

        gap_without = abs(
            no_penalty.search("injection", 2)[0].score - no_penalty.search("injection", 2)[1].score
        )
        gap_with = abs(
            full_penalty.search("injection", 2)[0].score
            - full_penalty.search("injection", 2)[1].score
        )
        assert gap_with > gap_without


class TestDense:
    @pytest.fixture
    def retriever(self) -> DenseRetriever:
        r = DenseRetriever(HashingEmbedder(dimensions=128))
        r.index(CORPUS)
        return r

    def test_returns_k_results(self, retriever: DenseRetriever) -> None:
        assert len(retriever.search("shell access", 3)) == 3

    def test_scores_are_descending(self, retriever: DenseRetriever) -> None:
        scores = [r.score for r in retriever.search("injection", 4)]
        assert scores == sorted(scores, reverse=True)

    def test_cosine_scores_stay_in_range(self, retriever: DenseRetriever) -> None:
        assert all(-1.01 <= r.score <= 1.01 for r in retriever.search("anything", 5))

    def test_exact_text_matches_itself_best(self, retriever: DenseRetriever) -> None:
        query = CORPUS[0].text
        assert retriever.search(query, 1)[0].chunk.chunk_id == "ssh"

    def test_searching_before_indexing_raises(self) -> None:
        with pytest.raises(RuntimeError, match="index\\(\\) must be called"):
            DenseRetriever(HashingEmbedder()).search("anything", 1)

    def test_k_larger_than_corpus(self, retriever: DenseRetriever) -> None:
        assert len(retriever.search("anything", 999)) == len(CORPUS)


class TestHybrid:
    def test_fuses_both_retrievers(self) -> None:
        hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(HashingEmbedder())])
        hybrid.index(CORPUS)
        assert len(hybrid.search("injection", 3)) == 3

    def test_rrf_scores_are_rank_based_not_score_based(self) -> None:
        """The defining property: with one retriever, the top result's score is
        exactly 1/(k+1), regardless of the underlying BM25 magnitude."""
        hybrid = HybridRetriever([BM25Retriever()], k=60)
        hybrid.index(CORPUS)
        assert hybrid.search("SSH port 22", 1)[0].score == pytest.approx(1 / 61)

    def test_agreement_between_retrievers_wins(self) -> None:
        """A document ranked second by both beats one ranked first by only one:
        2/(60+2) = 0.03226 > 1/(60+1) = 0.01639."""

        class Stub(BM25Retriever):
            def __init__(self, order: list[str]) -> None:
                super().__init__()
                self.order = order

            def index(self, chunks: list[Chunk]) -> None:
                self._by_id = {c.chunk_id: c for c in chunks}

            def search(self, query: str, k: int) -> list[ScoredChunk]:
                return [ScoredChunk(chunk=self._by_id[cid], score=1.0) for cid in self.order[:k]]

        hybrid = HybridRetriever([Stub(["ssh", "tls"]), Stub(["sql", "tls"])])
        hybrid.index(CORPUS)
        assert hybrid.search("q", 1)[0].chunk.chunk_id == "tls"

    def test_weights_shift_the_balance(self) -> None:
        bm25_only = HybridRetriever(
            [BM25Retriever(), DenseRetriever(HashingEmbedder())], weights=[1.0, 0.0]
        )
        bm25_only.index(CORPUS)
        assert bm25_only.search("SSH port 22", 1)[0].chunk.chunk_id == "ssh"

    def test_rejects_empty_retriever_list(self) -> None:
        with pytest.raises(ValueError, match="at least one retriever"):
            HybridRetriever([])

    def test_rejects_mismatched_weights(self) -> None:
        with pytest.raises(ValueError, match="weights must match"):
            HybridRetriever([BM25Retriever()], weights=[1.0, 1.0])


class TestDocumentDeduplication:
    def test_multiple_chunks_of_one_document_count_once(self) -> None:
        """Metrics are scored per document, so three chunks of the same file
        must not consume three of the k slots."""
        chunks = [
            chunk("doc-a#0", "injection injection", doc_id="doc-a"),
            chunk("doc-a#1", "injection input", doc_id="doc-a"),
            chunk("doc-b#0", "injection shell", doc_id="doc-b"),
        ]
        retriever = BM25Retriever()
        retriever.index(chunks)
        assert retriever.search_documents("injection", 2) == ["doc-a", "doc-b"]

    def test_rank_order_is_preserved(self) -> None:
        retriever = BM25Retriever()
        retriever.index(CORPUS)
        documents = retriever.search_documents("SSH port 22", 3)
        assert documents[0] == "ssh"
