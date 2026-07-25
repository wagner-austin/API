"""Tests for the assembled quality metrics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.benchmarking.quality import compute_quality
from covenant_ml.benchmarking.types import ERR_LENGTH_MISMATCH


def test_perfect_separation_scores_at_the_ceiling() -> None:
    labels: list[int] = [0, 0, 1, 1]
    scores: list[float] = [0.01, 0.02, 0.98, 0.99]
    y_true: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)
    proba: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    metrics = compute_quality(y_true, proba)
    assert metrics["auc_roc"] == 1.0
    assert metrics["auc_pr"] == 1.0
    assert metrics["brier"] < 0.01


def test_rates_are_computed_from_the_arrays() -> None:
    labels: list[int] = [0, 0, 0, 1]
    scores: list[float] = [0.2, 0.2, 0.2, 0.2]
    y_true: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)
    proba: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    metrics = compute_quality(y_true, proba)
    assert metrics["positive_rate"] == 0.25
    assert metrics["mean_pred"] == pytest.approx(0.2)


def test_mismatched_lengths_raise() -> None:
    labels: list[int] = [0, 1]
    scores: list[float] = [0.5]
    y_true: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)
    proba: NDArray[np.float64] = np.asarray(scores, dtype=np.float64)
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        compute_quality(y_true, proba)
