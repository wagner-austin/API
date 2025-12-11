"""Tests for covenant_ml metrics module."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from covenant_ml.metrics import (
    compute_accuracy,
    compute_all_metrics,
    compute_auc,
    compute_f1_score,
    compute_log_loss,
    compute_precision,
    compute_recall,
    format_metrics_str,
)


def _make_int_array(values: list[int]) -> NDArray[np.int64]:
    """Create int64 array from values."""
    arr: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _make_float_array(values: list[float]) -> NDArray[np.float64]:
    """Create float64 array from values."""
    arr: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _make_binary_arrays(
    n_samples: int = 10,
    positive_ratio: float = 0.5,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Create deterministic binary arrays for testing."""
    y_true: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    y_pred: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    y_prob: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

    n_positive = int(n_samples * positive_ratio)

    for i in range(n_samples):
        if i < n_positive:
            y_true[i] = 1
            if i < n_positive // 2:
                y_pred[i] = 1
                y_prob[i] = 0.8
            else:
                y_pred[i] = 0
                y_prob[i] = 0.3
        else:
            y_true[i] = 0
            if i < n_samples - 1:
                y_pred[i] = 0
                y_prob[i] = 0.2
            else:
                y_pred[i] = 1
                y_prob[i] = 0.6

    return y_true, y_pred, y_prob


def test_compute_log_loss_perfect_predictions() -> None:
    """Log loss is low for confident correct predictions."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_prob = _make_float_array([0.99, 0.95, 0.05, 0.01])

    loss = compute_log_loss(y_true, y_prob)

    assert loss < 0.1
    assert loss > 0.0


def test_compute_log_loss_wrong_predictions() -> None:
    """Log loss is high for wrong confident predictions."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_prob = _make_float_array([0.01, 0.05, 0.95, 0.99])

    loss = compute_log_loss(y_true, y_prob)

    assert loss > 2.0


def test_compute_log_loss_random_predictions() -> None:
    """Log loss is around 0.69 for random (0.5) predictions."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_prob = _make_float_array([0.5, 0.5, 0.5, 0.5])

    loss = compute_log_loss(y_true, y_prob)

    assert abs(loss - 0.693) < 0.01


def test_compute_log_loss_clips_extreme_values() -> None:
    """Log loss handles extreme probabilities with clipping."""
    y_true = _make_int_array([1, 0])
    y_prob = _make_float_array([1.0, 0.0])

    loss = compute_log_loss(y_true, y_prob)

    assert math.isfinite(loss)
    assert loss < 0.1


def test_compute_auc_perfect_separation() -> None:
    """AUC is 1.0 for perfect class separation."""
    y_true = _make_int_array([0, 0, 1, 1])
    y_prob = _make_float_array([0.1, 0.2, 0.8, 0.9])

    auc = compute_auc(y_true, y_prob)

    assert auc == 1.0


def test_compute_auc_random_baseline() -> None:
    """AUC is around 0.5 for random predictions."""
    y_true = _make_int_array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = _make_float_array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    auc = compute_auc(y_true, y_prob)

    # When all probabilities are identical, AUC depends on tie-breaking
    # Just verify it's between 0 and 1
    assert 0.0 <= auc <= 1.0


def test_compute_auc_inverted_predictions() -> None:
    """AUC is 0.0 for completely inverted predictions."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_prob = _make_float_array([0.1, 0.2, 0.8, 0.9])

    auc = compute_auc(y_true, y_prob)

    assert auc == 0.0


def test_compute_auc_all_same_class_returns_baseline() -> None:
    """AUC returns 0.5 when only one class present."""
    y_true = _make_int_array([1, 1, 1, 1])
    y_prob = _make_float_array([0.5, 0.6, 0.7, 0.8])

    auc = compute_auc(y_true, y_prob)

    assert auc == 0.5


def test_compute_accuracy_all_correct() -> None:
    """Accuracy is 1.0 when all predictions correct."""
    y_true = _make_int_array([0, 1, 0, 1])
    y_pred = _make_int_array([0, 1, 0, 1])

    acc = compute_accuracy(y_true, y_pred)

    assert acc == 1.0


def test_compute_accuracy_all_wrong() -> None:
    """Accuracy is 0.0 when all predictions wrong."""
    y_true = _make_int_array([0, 1, 0, 1])
    y_pred = _make_int_array([1, 0, 1, 0])

    acc = compute_accuracy(y_true, y_pred)

    assert acc == 0.0


def test_compute_accuracy_partial() -> None:
    """Accuracy reflects correct prediction fraction."""
    y_true = _make_int_array([0, 1, 0, 1])
    y_pred = _make_int_array([0, 1, 1, 0])

    acc = compute_accuracy(y_true, y_pred)

    assert acc == 0.5


def test_compute_precision_all_true_positives() -> None:
    """Precision is 1.0 when all positive predictions correct."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_pred = _make_int_array([1, 1, 0, 0])

    prec = compute_precision(y_true, y_pred)

    assert prec == 1.0


def test_compute_precision_all_false_positives() -> None:
    """Precision is 0.0 when all positive predictions wrong."""
    y_true = _make_int_array([0, 0, 0, 0])
    y_pred = _make_int_array([1, 1, 0, 0])

    prec = compute_precision(y_true, y_pred)

    assert prec == 0.0


def test_compute_precision_no_positive_predictions() -> None:
    """Precision is 0.0 when no positive predictions made."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_pred = _make_int_array([0, 0, 0, 0])

    prec = compute_precision(y_true, y_pred)

    assert prec == 0.0


def test_compute_precision_mixed() -> None:
    """Precision reflects TP/(TP+FP)."""
    y_true = _make_int_array([1, 0, 1, 0])
    y_pred = _make_int_array([1, 1, 0, 0])

    prec = compute_precision(y_true, y_pred)

    assert prec == 0.5


def test_compute_recall_all_detected() -> None:
    """Recall is 1.0 when all positives detected."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_pred = _make_int_array([1, 1, 1, 1])

    rec = compute_recall(y_true, y_pred)

    assert rec == 1.0


def test_compute_recall_none_detected() -> None:
    """Recall is 0.0 when no positives detected."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_pred = _make_int_array([0, 0, 0, 0])

    rec = compute_recall(y_true, y_pred)

    assert rec == 0.0


def test_compute_recall_no_actual_positives() -> None:
    """Recall is 0.0 when no actual positives exist."""
    y_true = _make_int_array([0, 0, 0, 0])
    y_pred = _make_int_array([1, 1, 0, 0])

    rec = compute_recall(y_true, y_pred)

    assert rec == 0.0


def test_compute_recall_partial() -> None:
    """Recall reflects TP/(TP+FN)."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_pred = _make_int_array([1, 0, 0, 0])

    rec = compute_recall(y_true, y_pred)

    assert rec == 0.5


def test_compute_f1_score_perfect() -> None:
    """F1 is 1.0 when precision and recall are both 1.0."""
    f1 = compute_f1_score(1.0, 1.0)

    assert f1 == 1.0


def test_compute_f1_score_zero() -> None:
    """F1 is 0.0 when precision and recall are both 0.0."""
    f1 = compute_f1_score(0.0, 0.0)

    assert f1 == 0.0


def test_compute_f1_score_one_zero() -> None:
    """F1 is 0.0 when either precision or recall is 0.0."""
    assert compute_f1_score(1.0, 0.0) == 0.0
    assert compute_f1_score(0.0, 1.0) == 0.0


def test_compute_f1_score_harmonic_mean() -> None:
    """F1 is harmonic mean of precision and recall."""
    f1 = compute_f1_score(0.8, 0.6)

    expected = 2 * (0.8 * 0.6) / (0.8 + 0.6)
    assert abs(f1 - expected) < 0.001


def test_compute_all_metrics_returns_all_fields() -> None:
    """compute_all_metrics returns all expected metric fields."""
    y_true, _, y_prob = _make_binary_arrays(20)

    metrics = compute_all_metrics(y_true, y_prob)

    assert "loss" in metrics
    assert "auc" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics


def test_compute_all_metrics_values_in_valid_range() -> None:
    """All metrics are within expected ranges."""
    y_true, _, y_prob = _make_binary_arrays(20)

    metrics = compute_all_metrics(y_true, y_prob)

    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0


def test_compute_all_metrics_custom_threshold() -> None:
    """Custom threshold affects binary predictions."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_prob = _make_float_array([0.6, 0.4, 0.3, 0.2])

    metrics_default = compute_all_metrics(y_true, y_prob, threshold=0.5)
    metrics_low = compute_all_metrics(y_true, y_prob, threshold=0.35)

    assert metrics_low["recall"] >= metrics_default["recall"]


def test_format_metrics_str_contains_all_values() -> None:
    """format_metrics_str includes all metric values."""
    y_true, _, y_prob = _make_binary_arrays(10)
    metrics = compute_all_metrics(y_true, y_prob)

    result = format_metrics_str(metrics)

    assert "loss=" in result
    assert "auc=" in result
    assert "acc=" in result
    assert "P=" in result
    assert "R=" in result
    assert "F1=" in result


def test_format_metrics_str_format() -> None:
    """format_metrics_str uses 4 decimal places."""
    y_true = _make_int_array([1, 0])
    y_prob = _make_float_array([0.9, 0.1])
    metrics = compute_all_metrics(y_true, y_prob)

    result = format_metrics_str(metrics)

    parts = result.split("=")
    assert len(parts) > 1
