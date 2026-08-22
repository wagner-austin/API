"""Tests for covenant_ml metrics module."""

from __future__ import annotations

import math

from covenant_ml.metrics_regression import (
    compute_all_regression_metrics,
    compute_mae,
    compute_mape,
    compute_mse,
    compute_r_squared,
    compute_rmse,
    format_regression_metrics_str,
)
from tests._metrics_fixtures import (
    _make_float_array,
)


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
