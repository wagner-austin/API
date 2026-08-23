"""Tests for the multiclass log-loss metric."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.metrics import compute_multiclass_log_loss


def _labels(values: list[int]) -> NDArray[np.int64]:
    """Build an int64 label vector.

    Args:
        values: Label values.

    Returns:
        The typed array.
    """
    return np.asarray(values, dtype=np.int64)


def _proba(rows: list[list[float]]) -> NDArray[np.float64]:
    """Build a float64 probability matrix.

    Args:
        rows: Probability rows.

    Returns:
        The typed array.
    """
    return np.asarray(rows, dtype=np.float64)


class TestComputeMulticlassLogLoss:
    """The metric matches its closed-form values and rejects bad shapes."""

    def test_perfect_predictions_score_near_zero(self) -> None:
        """Probability 1 on the true class gives loss at the clip floor."""
        y = _labels([0, 1, 2])
        p = _proba([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        loss = compute_multiclass_log_loss(y, p)
        assert loss == pytest.approx(-math.log(1.0 - 1e-15), abs=1e-12)

    def test_uniform_predictions_score_log_k(self) -> None:
        """The uniform distribution over K classes scores exactly log(K)."""
        third = 1.0 / 3.0
        y = _labels([0, 1, 2])
        p = _proba([[third] * 3, [third] * 3, [third] * 3])
        loss = compute_multiclass_log_loss(y, p)
        assert loss == pytest.approx(math.log(3.0), rel=1e-12)

    def test_matches_the_hand_computed_mean(self) -> None:
        """The loss is the mean negative log of the true-class probabilities."""
        y = _labels([1, 0])
        p = _proba([[0.2, 0.8], [0.6, 0.4]])
        expected = -(math.log(0.8) + math.log(0.6)) / 2.0
        assert compute_multiclass_log_loss(y, p) == pytest.approx(expected, rel=1e-12)

    def test_zero_probability_is_clipped_not_infinite(self) -> None:
        """A zero on the true class clips to eps instead of diverging."""
        y = _labels([0])
        p = _proba([[0.0, 1.0]])
        loss = compute_multiclass_log_loss(y, p)
        assert loss == pytest.approx(-math.log(1e-15), rel=1e-12)

    def test_rejects_a_length_mismatch(self) -> None:
        """Label count must match the probability rows."""
        y = _labels([0, 1, 0])
        p = _proba([[0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="equal length, got 3 and 2"):
            compute_multiclass_log_loss(y, p)

    def test_rejects_an_out_of_range_label(self) -> None:
        """A label past the probability columns names itself and its index."""
        y = _labels([0, 2])
        p = _proba([[0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match=r"label 2 at index 1 outside \[0, 2\)"):
            compute_multiclass_log_loss(y, p)

    def test_rejects_a_negative_label(self) -> None:
        """A negative label is outside the class range."""
        y = _labels([-1])
        p = _proba([[0.5, 0.5]])
        with pytest.raises(ValueError, match=r"label -1 at index 0 outside \[0, 2\)"):
            compute_multiclass_log_loss(y, p)
