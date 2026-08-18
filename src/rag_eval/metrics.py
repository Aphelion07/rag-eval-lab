"""Retrieval metrics.

Four numbers, because no single one is enough:

* **Recall@k**    - did the relevant documents make it into the top k at all?
  This is the ceiling on everything downstream: a generator cannot cite what
  retrieval never returned.
* **Precision@k** - how much of the top k is noise? Noise costs context window
  and pushes the real answer further from the model's attention.
* **MRR**         - how high did the *first* relevant document rank? The metric
  that matters when the answer lives in one place.
* **nDCG@k**      - rank-weighted, and the only one here that understands
  graded relevance ("this document answers it" vs "this one mentions it").

Reporting only Recall@k is the most common way to make a retriever look better
than it is, which is why all four are computed for every run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of relevant documents appearing in the top k."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the top k that is relevant.

    Divides by ``k`` rather than by the number retrieved: a retriever that
    returns three documents when ten were asked for has not earned the higher
    score that dividing by three would give it.
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document, or 0 if none was retrieved."""
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant document is in the top k. The blunt question:
    could this query have been answered at all?"""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the standard log2(rank + 1) discount."""
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    """Normalised DCG over graded relevance.

    ``relevance`` maps document id to a gain; documents absent from it score 0.
    Normalising against the ideal ordering is what makes scores comparable
    across queries that have different numbers of relevant documents.
    """
    if not relevance:
        return 0.0
    gains = [relevance.get(doc_id, 0.0) for doc_id in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    A benchmark over 40 queries produces a noisy mean. Without an interval,
    "hybrid scored 0.81 and dense scored 0.79" reads as a result when it may
    well be sampling noise. This is what lets the report say *whether* a
    difference is real.
    """
    if not values:
        return (0.0, 0.0)

    import random

    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    lower_index = int((1 - confidence) / 2 * resamples)
    upper_index = int((1 + confidence) / 2 * resamples) - 1
    return (means[lower_index], means[max(lower_index, upper_index)])


__all__ = [
    "bootstrap_ci",
    "dcg",
    "hit_at_k",
    "mean",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
