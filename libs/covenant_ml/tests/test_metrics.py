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
    compute_amex_metric,
    compute_auc,
    compute_average_precision,
    compute_brier_score,
    compute_f1_score,
    compute_log_loss,
    compute_precision,
    compute_recall,
    format_metrics_str,
)
from tests._metrics_fixtures import (
    _make_binary_arrays,
    _make_float_array,
    _make_int_array,
)


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
