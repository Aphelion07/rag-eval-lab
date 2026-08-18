"""End-to-end runs, aggregation, and the report rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_eval.chunking import ParagraphChunker, SentenceChunker
from rag_eval.corpus import Document, Query
from rag_eval.embedding import HashingEmbedder
from rag_eval.experiment import Config, run_config, run_grid
from rag_eval.report import (
    bar,
    chart,
    failure_analysis,
    full_report,
    leaderboard,
    significance_note,
    write_json,
)
from rag_eval.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

DOCUMENTS = [
    Document(doc_id="ssh", title="SSH", text="SSH uses TCP port 22 for encrypted shell access."),
    Document(doc_id="tls", title="TLS", text="TLS authenticates a server using certificates."),
    Document(doc_id="sql", title="SQL Injection", text="Untrusted input changes a query."),
    Document(doc_id="xss", title="XSS", text="Attacker script runs in another user's browser."),
]

QUERIES = [
    Query(query_id="q1", text="What port does SSH use?", relevance={"ssh": 2.0}),
    Query(query_id="q2", text="How does TLS authenticate?", relevance={"tls": 2.0}),
    Query(query_id="q3", text="What is SQL injection?", relevance={"sql": 2.0, "xss": 1.0}),
]


@pytest.fixture
def bm25_config() -> Config:
    return Config(SentenceChunker(size=256, overlap=32), BM25Retriever, "sentence+bm25")


class TestRunConfig:
    def test_scores_every_query(self, bm25_config: Config) -> None:
        result = run_config(bm25_config, DOCUMENTS, QUERIES, k=3)
        assert len(result.queries) == len(QUERIES)

    def test_records_index_and_query_timing(self, bm25_config: Config) -> None:
        result = run_config(bm25_config, DOCUMENTS, QUERIES, k=3)
        assert result.index_seconds >= 0
        assert result.latency_p50_ms >= 0
        assert result.chunk_count > 0

    def test_metrics_are_in_range(self, bm25_config: Config) -> None:
        result = run_config(bm25_config, DOCUMENTS, QUERIES, k=3)
        for value in (result.recall, result.precision, result.mrr, result.ndcg, result.hit_rate):
            assert 0.0 <= value <= 1.0

    def test_finds_the_obvious_answers(self, bm25_config: Config) -> None:
        """A sanity floor: on a corpus this easy, lexical retrieval should not
        be missing anything."""
        result = run_config(bm25_config, DOCUMENTS, QUERIES, k=3)
        assert result.hit_rate == 1.0

    def test_confidence_interval_brackets_the_mean(self, bm25_config: Config) -> None:
        result = run_config(bm25_config, DOCUMENTS, QUERIES, k=3)
        low, high = result.recall_ci
        assert low <= result.recall <= high

    def test_failures_lists_missed_queries(self) -> None:
        impossible = [Query(query_id="q9", text="zzz nothing", relevance={"ssh": 2.0})]
        config = Config(ParagraphChunker(), BM25Retriever, "paragraph+bm25")
        result = run_config(config, DOCUMENTS, impossible, k=3)
        assert result.failures(impossible) == [("q9", "zzz nothing")]

    def test_each_run_gets_a_fresh_retriever(self) -> None:
        """Configs hold a factory, not an instance. Sharing one would carry an
        index from the previous configuration into the next."""
        config = Config(ParagraphChunker(), BM25Retriever, "paragraph+bm25")
        first = run_config(config, DOCUMENTS, QUERIES, k=3)
        second = run_config(config, DOCUMENTS[:2], QUERIES, k=3)
        assert first.chunk_count != second.chunk_count


class TestRunGrid:
    @pytest.fixture
    def configs(self) -> list[Config]:
        embedder = HashingEmbedder(dimensions=64)
        chunker = ParagraphChunker()
        return [
            Config(chunker, BM25Retriever, "paragraph+bm25"),
            Config(chunker, lambda: DenseRetriever(embedder), "paragraph+dense"),
            Config(
                chunker,
                lambda: HybridRetriever([BM25Retriever(), DenseRetriever(embedder)]),
                "paragraph+hybrid",
            ),
        ]

    def test_runs_every_configuration(self, configs: list[Config]) -> None:
        results = run_grid(configs, DOCUMENTS, QUERIES, k=3)
        assert [r.label for r in results] == [c.label for c in configs]

    def test_progress_callback_fires_per_config(self, configs: list[Config]) -> None:
        seen: list[str] = []
        run_grid(configs, DOCUMENTS, QUERIES, k=3, on_progress=lambda label, _: seen.append(label))
        assert len(seen) == len(configs)


class TestReport:
    @pytest.fixture
    def results(self) -> list:
        embedder = HashingEmbedder(dimensions=64)
        return run_grid(
            [
                Config(ParagraphChunker(), BM25Retriever, "paragraph+bm25"),
                Config(ParagraphChunker(), lambda: DenseRetriever(embedder), "paragraph+dense"),
            ],
            DOCUMENTS,
            QUERIES,
            k=3,
        )

    def test_bar_endpoints(self) -> None:
        assert bar(0.0).strip(".") == ""
        assert set(bar(1.0)) == {"#"}

    def test_leaderboard_is_a_markdown_table(self, results: list) -> None:
        table = leaderboard(results)
        assert table.startswith("| # |")
        assert table.count("\n") == len(results) + 1  # header plus separator

    def test_leaderboard_is_sorted_by_the_metric(self, results: list) -> None:
        rows = leaderboard(results).splitlines()[2:]
        ranked = sorted(results, key=lambda r: r.ndcg, reverse=True)
        assert ranked[0].label in rows[0]

    def test_chart_lists_every_configuration(self, results: list) -> None:
        rendered = chart(results)
        assert all(r.label in rendered for r in results)

    def test_significance_note_states_a_verdict(self, results: list) -> None:
        note = significance_note(results)
        assert "overlap" in note

    def test_significance_note_needs_two_results(self, results: list) -> None:
        assert significance_note(results[:1]) == ""

    def test_failure_analysis_reports_a_clean_run(self, results: list) -> None:
        text = failure_analysis(results[0], QUERIES)
        assert "all 3 queries" in text or "found nothing relevant" in text

    def test_full_report_contains_every_section(self, results: list) -> None:
        report = full_report(results, QUERIES, k=3)
        for heading in ("Leaderboard", "Is the winner actually winning?", "still misses"):
            assert heading in report

    def test_write_json_round_trips(self, results: list, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "results.json"
        write_json(results, path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert len(rows) == len(results)
        assert {"config", "recall", "ndcg", "recall_ci"} <= set(rows[0])
