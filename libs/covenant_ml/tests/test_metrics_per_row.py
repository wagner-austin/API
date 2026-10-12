"""Tests for the per-row loss forms the paired evaluation differences."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from covenant_ml.metrics import (
    brier_per_row,
    compute_brier_score,
    compute_log_loss,
    log_loss_per_row,
)


def _floats(values: list[float]) -> NDArray[np.float64]:
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return array


def _labels() -> NDArray[np.int64]:
    values: list[int] = [1, 0, 1, 0]
    array: NDArray[np.int64] = np.asarray(values, dtype=np.int64)
    return array


def _proba() -> NDArray[np.float64]:
    return _floats([0.9, 0.2, 0.4, 0.6])


def test_log_loss_per_row_is_each_rows_cross_entropy() -> None:
    per_row = log_loss_per_row(_labels(), _proba())
    expected = _floats([-math.log(0.9), -math.log(0.8), -math.log(0.4), -math.log(0.4)])
    assert np.allclose(per_row, expected)


def test_log_loss_is_the_mean_of_its_rows() -> None:
    per_row = log_loss_per_row(_labels(), _proba())
    total = float(np.sum(per_row))
    assert compute_log_loss(_labels(), _proba()) == total / 4


def test_log_loss_per_row_clips_certain_wrong_answers() -> None:
    labels: NDArray[np.int64] = np.ones(1, dtype=np.int64)
    per_row = log_loss_per_row(labels, _floats([0.0]))
    assert float(per_row.flat[0].item()) == -math.log(1e-15)


def test_brier_per_row_is_each_rows_squared_error() -> None:
    per_row = brier_per_row(_labels(), _proba())
    assert np.allclose(per_row, _floats([0.01, 0.04, 0.36, 0.36]))


def test_brier_score_is_the_mean_of_its_rows() -> None:
    per_row = brier_per_row(_labels(), _proba())
    total = float(np.sum(per_row))
    assert compute_brier_score(_labels(), _proba()) == total / 4
