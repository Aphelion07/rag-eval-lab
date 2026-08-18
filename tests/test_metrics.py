"""Metric correctness, checked against hand-computed values.

Every expected number here is worked out by hand in a comment. A metric test
that computes the expectation the same way the implementation does proves only
that the code is consistent with itself.
"""

from __future__ import annotations

import pytest

from rag_eval.metrics import (
    bootstrap_ci,
    dcg,
    hit_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecall:
    def test_all_relevant_retrieved(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_half_retrieved(self) -> None:
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_relevant_below_cutoff_does_not_count(self) -> None:
        # 'b' sits at rank 3, outside k=2.
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, 2) == 0.5

    def test_no_relevant_labels_is_zero(self) -> None:
        assert recall_at_k(["a"], set(), 5) == 0.0

    def test_empty_retrieval_is_zero(self) -> None:
        assert recall_at_k([], {"a"}, 5) == 0.0


class TestPrecision:
    def test_divides_by_k_not_by_retrieved(self) -> None:
        """Returning 2 documents when 5 were asked for should not score as if
        only 2 were requested: 1 hit out of k=5 is 0.2, not 0.5."""
        assert precision_at_k(["a", "x"], {"a"}, 5) == pytest.approx(0.2)

    def test_perfect_precision(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_zero_k_is_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_only_the_first_hit_counts(self) -> None:
        # Relevant at ranks 2 and 3; only rank 2 contributes.
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_nothing_relevant_is_zero(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestHit:
    def test_any_relevant_scores_one(self) -> None:
        assert hit_at_k(["x", "y", "a"], {"a"}, 3) == 1.0

    def test_relevant_outside_k_scores_zero(self) -> None:
        assert hit_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


class TestNDCG:
    def test_perfect_ordering_is_one(self) -> None:
        relevance = {"a": 2.0, "b": 1.0}
        assert ndcg_at_k(["a", "b"], relevance, 2) == pytest.approx(1.0)

    def test_inverted_ordering_scores_below_one(self) -> None:
        """Same documents, wrong order.

        DCG  = 1/log2(2) + 2/log2(3) = 1.0 + 1.2619 = 2.2619
        IDCG = 2/log2(2) + 1/log2(3) = 2.0 + 0.6309 = 2.6309
        nDCG = 2.2619 / 2.6309 = 0.8598
        """
        relevance = {"a": 2.0, "b": 1.0}
        assert ndcg_at_k(["b", "a"], relevance, 2) == pytest.approx(0.8598, abs=1e-4)

    def test_grades_distinguish_partial_answers(self) -> None:
        """A document graded 2 ranked first beats one graded 1 ranked first,
        which is the whole reason for graded relevance."""
        relevance = {"good": 2.0, "partial": 1.0}
        strong = ndcg_at_k(["good"], relevance, 1)
        weak = ndcg_at_k(["partial"], relevance, 1)
        assert strong > weak

    def test_irrelevant_results_score_zero(self) -> None:
        assert ndcg_at_k(["x", "y"], {"a": 2.0}, 2) == 0.0

    def test_empty_relevance_is_zero(self) -> None:
        assert ndcg_at_k(["a"], {}, 5) == 0.0


class TestDCG:
    def test_known_series(self) -> None:
        # 3/log2(2) + 2/log2(3) + 1/log2(4) = 3.0 + 1.26186 + 0.5 = 4.76186
        assert dcg([3.0, 2.0, 1.0]) == pytest.approx(4.76186, abs=1e-5)

    def test_empty_is_zero(self) -> None:
        assert dcg([]) == 0.0


class TestBootstrap:
    def test_constant_series_has_zero_width_interval(self) -> None:
        low, high = bootstrap_ci([0.5] * 20)
        assert low == pytest.approx(0.5)
        assert high == pytest.approx(0.5)

    def test_interval_brackets_the_mean(self) -> None:
        values = [0.0, 0.25, 0.5, 0.75, 1.0] * 8
        low, high = bootstrap_ci(values)
        assert low <= mean(values) <= high

    def test_larger_sample_narrows_the_interval(self) -> None:
        """The property that makes the interval worth reporting: it should
        shrink as evidence accumulates."""
        pattern = [0.0, 1.0]
        small_low, small_high = bootstrap_ci(pattern * 10, seed=7)
        large_low, large_high = bootstrap_ci(pattern * 200, seed=7)
        assert (large_high - large_low) < (small_high - small_low)

    def test_is_deterministic_for_a_seed(self) -> None:
        values = [0.1, 0.9, 0.4, 0.6]
        assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)

    def test_empty_input(self) -> None:
        assert bootstrap_ci([]) == (0.0, 0.0)


class TestMean:
    def test_empty_is_zero(self) -> None:
        assert mean([]) == 0.0

    def test_average(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == 2.0
