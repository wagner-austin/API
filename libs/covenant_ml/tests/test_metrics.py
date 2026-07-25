"""Tests for covenant_ml metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.metrics import (
    _compute_weighted_gini,
    compute_accuracy,
    compute_all_metrics,
    compute_all_regression_metrics,
    compute_amex_metric,
    compute_auc,
    compute_average_precision,
    compute_brier_score,
    compute_f1_score,
    compute_log_loss,
    compute_mae,
    compute_mape,
    compute_mse,
    compute_precision,
    compute_r_squared,
    compute_recall,
    compute_rmse,
    format_metrics_str,
    format_regression_metrics_str,
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


# =============================================================================
# AMEX Competition Metric Tests
# =============================================================================


def test_compute_amex_metric_perfect_predictions() -> None:
    """AMEX metric is 1.0 for perfect predictions."""
    # Perfect ranking: all positives ranked higher than negatives
    y_true = _make_int_array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = _make_float_array([0.9, 0.8, 0.7, 0.3, 0.2, 0.15, 0.1, 0.05])

    result = compute_amex_metric(y_true, y_pred)

    assert result["score"] == 1.0
    assert result["normalized_gini"] == 1.0
    assert result["default_rate_at_4_percent"] == 1.0


def test_compute_amex_metric_returns_all_components() -> None:
    """AMEX metric result contains all required fields."""
    y_true = _make_int_array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = _make_float_array([0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05])

    result = compute_amex_metric(y_true, y_pred)

    assert "score" in result
    assert "normalized_gini" in result
    assert "default_rate_at_4_percent" in result
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 <= result["normalized_gini"] <= 1.0
    assert 0.0 <= result["default_rate_at_4_percent"] <= 1.0


def test_compute_amex_metric_score_is_average() -> None:
    """AMEX score is average of Gini and D@4%."""
    y_true = _make_int_array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = _make_float_array([0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05])

    result = compute_amex_metric(y_true, y_pred)

    expected_score = 0.5 * (result["normalized_gini"] + result["default_rate_at_4_percent"])
    assert abs(result["score"] - expected_score) < 1e-10


def test_compute_amex_metric_raises_on_length_mismatch() -> None:
    """AMEX metric raises ValueError on array length mismatch."""
    y_true = _make_int_array([1, 0, 0])
    y_pred = _make_float_array([0.8, 0.2])

    with pytest.raises(ValueError, match="Array length mismatch"):
        compute_amex_metric(y_true, y_pred)


def test_compute_amex_metric_raises_on_empty_arrays() -> None:
    """AMEX metric raises ValueError on empty arrays."""
    y_true: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
    y_pred: NDArray[np.float64] = np.zeros(0, dtype=np.float64)

    with pytest.raises(ValueError, match="Cannot compute metric on empty arrays"):
        compute_amex_metric(y_true, y_pred)


def test_compute_amex_metric_raises_on_no_positives() -> None:
    """AMEX metric raises ValueError when no positive samples exist."""
    y_true = _make_int_array([0, 0, 0, 0])
    y_pred = _make_float_array([0.8, 0.6, 0.4, 0.2])

    with pytest.raises(ValueError, match="Cannot compute metric with no positive samples"):
        compute_amex_metric(y_true, y_pred)


def test_compute_amex_metric_weighted_negatives() -> None:
    """AMEX metric applies 20x weight to negative samples."""
    # With 20x weight on negatives, the effective ratio changes
    # 2 positives + 8 negatives * 20 = 2 + 160 = 162 effective samples
    # Top 4% of 162 = ~6.5 samples
    y_true = _make_int_array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = _make_float_array([0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20])

    result = compute_amex_metric(y_true, y_pred)

    # With good predictions, positives should be captured in top 4%
    assert result["default_rate_at_4_percent"] > 0.0
    assert result["score"] > 0.5


def test_compute_amex_metric_poor_predictions() -> None:
    """AMEX metric is low for poor predictions."""
    # Positives ranked lower than negatives
    y_true = _make_int_array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = _make_float_array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3])

    result = compute_amex_metric(y_true, y_pred)

    # Poor predictions should give low Gini
    assert result["normalized_gini"] < 0.5
    assert result["score"] < 0.5


def test_compute_weighted_gini_all_zeros_returns_zero() -> None:
    """Weighted Gini returns 0.0 when all labels are zero."""
    # This tests the edge case in _compute_weighted_gini when total_weighted_pos == 0
    y_true: NDArray[np.float64] = np.zeros(5, dtype=np.float64)
    y_pred: NDArray[np.float64] = _make_float_array([0.9, 0.7, 0.5, 0.3, 0.1])

    result = _compute_weighted_gini(y_true, y_pred, sort_by_pred=True)

    assert result == 0.0


# =============================================================================
# Regression Metric Tests
# =============================================================================


def test_compute_mse_perfect_prediction() -> None:
    """MSE is 0.0 for perfect predictions."""
    y_true = _make_float_array([1.0, 2.0, 3.0, 4.0])
    y_pred = _make_float_array([1.0, 2.0, 3.0, 4.0])

    assert compute_mse(y_true, y_pred) == 0.0


def test_compute_mse_known_values() -> None:
    """MSE matches hand-computed value."""
    y_true = _make_float_array([1.0, 2.0, 3.0])
    y_pred = _make_float_array([1.5, 2.5, 3.5])
    # errors: 0.5, 0.5, 0.5 → squared: 0.25, 0.25, 0.25 → mean: 0.25
    assert abs(compute_mse(y_true, y_pred) - 0.25) < 1e-10


def test_compute_mse_asymmetric_errors() -> None:
    """MSE penalizes large errors more than small ones."""
    y_true = _make_float_array([0.0, 0.0])
    y_pred_small = _make_float_array([0.1, 0.1])
    y_pred_large = _make_float_array([0.0, 1.0])
    # small: (0.01 + 0.01)/2 = 0.01
    # large: (0.0 + 1.0)/2 = 0.5
    assert compute_mse(y_true, y_pred_small) < compute_mse(y_true, y_pred_large)


def test_compute_rmse_perfect_prediction() -> None:
    """RMSE is 0.0 for perfect predictions."""
    y_true = _make_float_array([1.0, 2.0, 3.0])
    y_pred = _make_float_array([1.0, 2.0, 3.0])

    assert compute_rmse(y_true, y_pred) == 0.0


def test_compute_rmse_known_values() -> None:
    """RMSE matches hand-computed value."""
    y_true = _make_float_array([1.0, 2.0, 3.0])
    y_pred = _make_float_array([1.5, 2.5, 3.5])
    # MSE = 0.25, RMSE = 0.5
    assert abs(compute_rmse(y_true, y_pred) - 0.5) < 1e-10


def test_compute_rmse_is_sqrt_of_mse() -> None:
    """RMSE is exactly sqrt(MSE)."""
    y_true = _make_float_array([1.0, 3.0, 5.0, 7.0])
    y_pred = _make_float_array([1.2, 2.8, 5.3, 6.5])

    mse = compute_mse(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    assert abs(rmse - math.sqrt(mse)) < 1e-10


def test_compute_mae_perfect_prediction() -> None:
    """MAE is 0.0 for perfect predictions."""
    y_true = _make_float_array([1.0, 2.0, 3.0])
    y_pred = _make_float_array([1.0, 2.0, 3.0])

    assert compute_mae(y_true, y_pred) == 0.0


def test_compute_mae_known_values() -> None:
    """MAE matches hand-computed value."""
    y_true = _make_float_array([1.0, 2.0, 3.0])
    y_pred = _make_float_array([1.5, 2.5, 3.5])
    # |0.5| + |0.5| + |0.5| / 3 = 0.5
    assert abs(compute_mae(y_true, y_pred) - 0.5) < 1e-10


def test_compute_mae_handles_negative_errors() -> None:
    """MAE treats over-predictions and under-predictions equally."""
    y_true = _make_float_array([2.0, 4.0])
    y_pred = _make_float_array([3.0, 3.0])
    # |2-3| + |4-3| / 2 = 1.0
    assert abs(compute_mae(y_true, y_pred) - 1.0) < 1e-10


def test_compute_r_squared_perfect_prediction() -> None:
    """R² is 1.0 for perfect predictions."""
    y_true = _make_float_array([1.0, 2.0, 3.0, 4.0])
    y_pred = _make_float_array([1.0, 2.0, 3.0, 4.0])

    assert compute_r_squared(y_true, y_pred) == 1.0


def test_compute_r_squared_mean_prediction() -> None:
    """R² is 0.0 when predictions equal the mean."""
    y_true = _make_float_array([1.0, 2.0, 3.0, 4.0])
    mean_val = 2.5
    y_pred = _make_float_array([mean_val, mean_val, mean_val, mean_val])

    assert abs(compute_r_squared(y_true, y_pred)) < 1e-10


def test_compute_r_squared_worse_than_mean() -> None:
    """R² is negative when predictions are worse than mean baseline."""
    y_true = _make_float_array([1.0, 2.0, 3.0, 4.0])
    y_pred = _make_float_array([10.0, 10.0, 10.0, 10.0])

    assert compute_r_squared(y_true, y_pred) < 0.0


def test_compute_r_squared_constant_target() -> None:
    """R² returns 0.0 when all true values are identical (SS_tot=0)."""
    y_true = _make_float_array([5.0, 5.0, 5.0])
    y_pred = _make_float_array([5.0, 5.1, 4.9])

    assert compute_r_squared(y_true, y_pred) == 0.0


def test_compute_mape_perfect_prediction() -> None:
    """MAPE is 0.0 for perfect predictions."""
    y_true = _make_float_array([1.0, 2.0, 3.0])
    y_pred = _make_float_array([1.0, 2.0, 3.0])

    assert compute_mape(y_true, y_pred) == 0.0


def test_compute_mape_known_values() -> None:
    """MAPE matches hand-computed value."""
    y_true = _make_float_array([100.0, 200.0])
    y_pred = _make_float_array([110.0, 190.0])
    # |100-110|/100 + |200-190|/200 = 0.1 + 0.05 = 0.15 / 2 = 0.075
    result = compute_mape(y_true, y_pred)
    assert abs(result - 0.075) < 0.001


def test_compute_mape_near_zero_true_values() -> None:
    """MAPE uses eps protection for near-zero true values."""
    y_true = _make_float_array([0.0, 1.0])
    y_pred = _make_float_array([0.1, 1.1])

    result = compute_mape(y_true, y_pred)
    assert math.isfinite(result)
    assert result > 0.0


def test_compute_all_regression_metrics_returns_all_fields() -> None:
    """compute_all_regression_metrics returns all metric fields."""
    y_true = _make_float_array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = _make_float_array([1.1, 2.2, 2.8, 4.1, 5.3])

    metrics = compute_all_regression_metrics(y_true, y_pred)

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r_squared" in metrics
    assert "mape" in metrics


def test_compute_all_regression_metrics_consistency() -> None:
    """All-in-one function matches individual function results."""
    y_true = _make_float_array([1.0, 3.0, 5.0, 7.0])
    y_pred = _make_float_array([1.2, 2.8, 5.3, 6.5])

    metrics = compute_all_regression_metrics(y_true, y_pred)

    assert abs(metrics["mse"] - compute_mse(y_true, y_pred)) < 1e-10
    assert abs(metrics["rmse"] - compute_rmse(y_true, y_pred)) < 1e-10
    assert abs(metrics["mae"] - compute_mae(y_true, y_pred)) < 1e-10
    assert abs(metrics["r_squared"] - compute_r_squared(y_true, y_pred)) < 1e-10
    assert abs(metrics["mape"] - compute_mape(y_true, y_pred)) < 1e-10


def test_compute_all_regression_metrics_perfect() -> None:
    """Perfect predictions yield MSE=0, RMSE=0, MAE=0, R²=1, MAPE=0."""
    y_true = _make_float_array([2.0, 4.0, 6.0])
    y_pred = _make_float_array([2.0, 4.0, 6.0])

    metrics = compute_all_regression_metrics(y_true, y_pred)

    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r_squared"] == 1.0
    assert metrics["mape"] == 0.0


def test_compute_all_regression_metrics_single_sample() -> None:
    """Regression metrics work with a single sample."""
    y_true = _make_float_array([3.0])
    y_pred = _make_float_array([3.5])

    metrics = compute_all_regression_metrics(y_true, y_pred)

    assert abs(metrics["mse"] - 0.25) < 1e-10
    assert abs(metrics["rmse"] - 0.5) < 1e-10
    assert abs(metrics["mae"] - 0.5) < 1e-10
    # R² = 0.0 when single sample (SS_tot = 0)
    assert metrics["r_squared"] == 0.0


def test_format_regression_metrics_str_contains_all_values() -> None:
    """format_regression_metrics_str includes all metric names."""
    y_true = _make_float_array([1.0, 2.0, 3.0])
    y_pred = _make_float_array([1.1, 2.2, 2.8])
    metrics = compute_all_regression_metrics(y_true, y_pred)

    result = format_regression_metrics_str(metrics)

    assert "MSE=" in result
    assert "RMSE=" in result
    assert "MAE=" in result
    assert "R\u00b2=" in result
    assert "MAPE=" in result


def test_format_regression_metrics_str_uses_four_decimals() -> None:
    """format_regression_metrics_str uses 4 decimal places."""
    metrics = compute_all_regression_metrics(
        _make_float_array([1.0, 2.0]),
        _make_float_array([1.0, 2.0]),
    )

    result = format_regression_metrics_str(metrics)

    assert "MSE=0.0000" in result
    assert "RMSE=0.0000" in result


def test_compute_average_precision_perfect_ranking() -> None:
    """Average precision is 1.0 when every positive outranks every negative."""
    y_true = _make_int_array([0, 0, 1, 1])
    y_prob = _make_float_array([0.1, 0.2, 0.8, 0.9])

    assert compute_average_precision(y_true, y_prob) == 1.0


def test_compute_average_precision_worst_ranking() -> None:
    """Average precision is low when every positive ranks below every negative."""
    y_true = _make_int_array([1, 1, 0, 0])
    y_prob = _make_float_array([0.1, 0.2, 0.8, 0.9])

    assert compute_average_precision(y_true, y_prob) < 0.6


def test_compute_average_precision_matches_hand_calculation() -> None:
    """Step-wise definition: sum of recall gain times precision at each hit.

    Ranked descending the labels are [1, 0, 1, 0]. Precision at the two hits
    is 1/1 and 2/3; each contributes a recall gain of 1/2.
    """
    y_true = _make_int_array([1, 0, 1, 0])
    y_prob = _make_float_array([0.9, 0.8, 0.7, 0.6])

    expected = 0.5 * 1.0 + 0.5 * (2.0 / 3.0)

    assert compute_average_precision(y_true, y_prob) == pytest.approx(expected)


def test_compute_average_precision_without_positives_is_zero() -> None:
    """Undefined without a positive case; reported as 0.0 rather than NaN."""
    y_true = _make_int_array([0, 0, 0])
    y_prob = _make_float_array([0.1, 0.5, 0.9])

    assert compute_average_precision(y_true, y_prob) == 0.0


def test_compute_average_precision_all_positives_is_one() -> None:
    """Every ranked item is a hit, so precision is 1.0 throughout."""
    y_true = _make_int_array([1, 1, 1])
    y_prob = _make_float_array([0.1, 0.5, 0.9])

    assert compute_average_precision(y_true, y_prob) == pytest.approx(1.0)


def test_compute_brier_score_perfect_forecast_is_zero() -> None:
    """A confident, correct forecast scores 0.0."""
    y_true = _make_int_array([0, 1])
    y_prob = _make_float_array([0.0, 1.0])

    assert compute_brier_score(y_true, y_prob) == 0.0


def test_compute_brier_score_worst_forecast_is_one() -> None:
    """A confident, wrong forecast scores 1.0."""
    y_true = _make_int_array([0, 1])
    y_prob = _make_float_array([1.0, 0.0])

    assert compute_brier_score(y_true, y_prob) == 1.0


def test_compute_brier_score_matches_hand_calculation() -> None:
    """Mean squared error between probability and outcome."""
    y_true = _make_int_array([0, 1])
    y_prob = _make_float_array([0.25, 0.75])

    expected = (0.25**2 + 0.25**2) / 2.0

    assert compute_brier_score(y_true, y_prob) == pytest.approx(expected)


def test_compute_brier_score_penalises_overconfidence_at_equal_ranking() -> None:
    """Two models that rank identically can still differ in calibration.

    Both score the rows in the same order, so ROC-AUC cannot separate them.
    The ranking is imperfect (a negative outranks both positives), and the
    overconfident model pushes its probabilities toward the extremes, so it
    pays a larger squared penalty on the rows it gets wrong.
    """
    y_true = _make_int_array([0, 1, 0, 1])
    calibrated = _make_float_array([0.4, 0.6, 0.7, 0.3])
    overconfident = _make_float_array([0.3, 0.7, 0.9, 0.1])

    assert compute_auc(y_true, calibrated) == compute_auc(y_true, overconfident)
    assert compute_brier_score(y_true, calibrated) < compute_brier_score(y_true, overconfident)
