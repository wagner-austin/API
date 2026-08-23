"""Tests for the NDCG@k metric."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.metrics import compute_ndcg_at_k


def _grades(values: list[int]) -> NDArray[np.int64]:
    """Build an int64 grade vector.

    Args:
        values: Grade values.

    Returns:
        The typed array.
    """
    return np.asarray(values, dtype=np.int64)


def _scores(values: list[float]) -> NDArray[np.float64]:
    """Build a float64 score vector.

    Args:
        values: Score values.

    Returns:
        The typed array.
    """
    return np.asarray(values, dtype=np.float64)


class TestComputeNdcgAtK:
    """The metric matches its closed-form values and rejects bad inputs."""

    def test_perfect_ordering_scores_one(self) -> None:
        """Scores descending with grades give exactly 1.0."""
        result = compute_ndcg_at_k(_grades([2, 1, 0]), _scores([3.0, 2.0, 1.0]), 3)
        assert result == pytest.approx(1.0, abs=1e-12)

    def test_reversed_ordering_matches_the_hand_computed_ratio(self) -> None:
        """A fully reversed ranking matches the closed-form DCG ratio."""
        # Observed: gain 0 at pos 0, gain 1 at pos 1, gain 3 at pos 2.
        observed = 1.0 / math.log2(3.0) + 3.0 / math.log2(4.0)
        ideal = 3.0 + 1.0 / math.log2(3.0)
        result = compute_ndcg_at_k(_grades([2, 1, 0]), _scores([1.0, 2.0, 3.0]), 3)
        assert result == pytest.approx(observed / ideal, rel=1e-12)

    def test_truncation_only_counts_the_top_k(self) -> None:
        """At k=1 only the top-scored document contributes."""
        # Top-scored doc has grade 1 (gain 1); the ideal top is grade 2
        # (gain 3).
        result = compute_ndcg_at_k(_grades([2, 1]), _scores([0.0, 1.0]), 1)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-12)

    def test_all_zero_grades_score_one(self) -> None:
        """A query with nothing to rank scores 1.0 by definition."""
        result = compute_ndcg_at_k(_grades([0, 0]), _scores([0.5, 0.1]), 2)
        assert result == pytest.approx(1.0, abs=0.0)

    def test_rejects_a_length_mismatch(self) -> None:
        """Grade and score vectors must align."""
        with pytest.raises(ValueError, match="equal length, got 2 and 3"):
            compute_ndcg_at_k(_grades([1, 0]), _scores([1.0, 2.0, 3.0]), 2)

    def test_rejects_a_zero_k(self) -> None:
        """The truncation position must be positive."""
        with pytest.raises(ValueError, match="k must be >= 1, got 0"):
            compute_ndcg_at_k(_grades([1]), _scores([1.0]), 0)

    def test_rejects_a_negative_grade(self) -> None:
        """Grades are non-negative by definition."""
        with pytest.raises(ValueError, match="grade -1 at index 1 must be >= 0"):
            compute_ndcg_at_k(_grades([1, -1]), _scores([1.0, 2.0]), 2)
