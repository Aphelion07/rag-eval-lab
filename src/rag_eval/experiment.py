"""The experiment runner: build a configuration, score it, aggregate honestly.

One run evaluates one (chunker, retriever) pair against the whole golden set and
produces both the retrieval metrics and the operational ones - index time, query
latency, chunk count. Both matter. A configuration that wins on Recall@5 by two
points while taking six times as long to query has not obviously won.

Aggregation reports a bootstrap confidence interval alongside every mean.
Forty queries is a small sample, and without an interval a two point difference
between configurations reads as a result when it is usually noise.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import metrics as M
from .chunking import Chunker
from .corpus import Document, Query
from .retrieval import Retriever


@dataclass(frozen=True)
class Config:
    """One point in the search space."""

    chunker: Chunker
    # A factory rather than an instance: every run needs a retriever with an
    # empty index, and reusing one across configurations would carry state.
    build_retriever: Callable[[], Retriever]
    label: str


@dataclass
class QueryResult:
    query_id: str
    retrieved: list[str]
    latency_ms: float
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float
    hit: float


@dataclass
class RunResult:
    """Everything measured for one configuration."""

    label: str
    chunker: str
    retriever: str
    k: int
    chunk_count: int
    mean_chunk_chars: float
    index_seconds: float
    queries: list[QueryResult] = field(default_factory=list)

    # -- aggregates ---------------------------------------------------------

    def _column(self, name: str) -> list[float]:
        return [getattr(q, name) for q in self.queries]

    @property
    def recall(self) -> float:
        return M.mean(self._column("recall"))

    @property
    def precision(self) -> float:
        return M.mean(self._column("precision"))

    @property
    def mrr(self) -> float:
        return M.mean(self._column("reciprocal_rank"))

    @property
    def ndcg(self) -> float:
        return M.mean(self._column("ndcg"))

    @property
    def hit_rate(self) -> float:
        return M.mean(self._column("hit"))

    @property
    def recall_ci(self) -> tuple[float, float]:
        return M.bootstrap_ci(self._column("recall"))

    @property
    def latency_p50_ms(self) -> float:
        latencies = sorted(self._column("latency_ms"))
        return latencies[len(latencies) // 2] if latencies else 0.0

    @property
    def latency_p95_ms(self) -> float:
        latencies = sorted(self._column("latency_ms"))
        if not latencies:
            return 0.0
        index = max(0, min(len(latencies) - 1, int(0.95 * len(latencies)) - 1))
        return latencies[index]

    def failures(self, queries: list[Query]) -> list[tuple[str, str]]:
        """Queries where nothing relevant was retrieved at all.

        The most useful output of an evaluation run: an aggregate says a
        configuration is worse, this says *on what*.
        """
        by_id = {q.query_id: q for q in queries}
        return [
            (result.query_id, by_id[result.query_id].text)
            for result in self.queries
            if result.hit == 0.0
        ]

    def row(self) -> dict[str, object]:
        low, high = self.recall_ci
        return {
            "config": self.label,
            "chunker": self.chunker,
            "retriever": self.retriever,
            "chunks": self.chunk_count,
            "recall": round(self.recall, 4),
            "recall_ci": [round(low, 4), round(high, 4)],
            "precision": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "hit_rate": round(self.hit_rate, 4),
            "index_s": round(self.index_seconds, 3),
            "query_p50_ms": round(self.latency_p50_ms, 2),
            "query_p95_ms": round(self.latency_p95_ms, 2),
        }


def run_config(
    config: Config,
    documents: list[Document],
    queries: list[Query],
    *,
    k: int = 5,
) -> RunResult:
    """Index the corpus under one configuration and score every query."""
    chunks = config.chunker.chunk_all(documents)
    retriever = config.build_retriever()

    started = time.perf_counter()
    retriever.index(chunks)
    index_seconds = time.perf_counter() - started

    result = RunResult(
        label=config.label,
        chunker=config.chunker.name,
        retriever=retriever.name,
        k=k,
        chunk_count=len(chunks),
        mean_chunk_chars=M.mean([float(len(c.text)) for c in chunks]),
        index_seconds=index_seconds,
    )

    for query in queries:
        query_started = time.perf_counter()
        retrieved = retriever.search_documents(query.text, k)
        latency_ms = (time.perf_counter() - query_started) * 1000

        relevant = query.relevant_ids
        result.queries.append(
            QueryResult(
                query_id=query.query_id,
                retrieved=retrieved,
                latency_ms=latency_ms,
                recall=M.recall_at_k(retrieved, relevant, k),
                precision=M.precision_at_k(retrieved, relevant, k),
                reciprocal_rank=M.reciprocal_rank(retrieved, relevant),
                ndcg=M.ndcg_at_k(retrieved, query.relevance, k),
                hit=M.hit_at_k(retrieved, relevant, k),
            )
        )
    return result


def run_grid(
    configs: list[Config],
    documents: list[Document],
    queries: list[Query],
    *,
    k: int = 5,
    on_progress: Callable[[str, RunResult], None] | None = None,
) -> list[RunResult]:
    results: list[RunResult] = []
    for config in configs:
        result = run_config(config, documents, queries, k=k)
        results.append(result)
        if on_progress is not None:
            on_progress(config.label, result)
    return results


__all__ = ["Config", "QueryResult", "RunResult", "run_config", "run_grid"]
