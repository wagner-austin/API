"""Tests for integrated gradients: ConstantGradientModel."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from platform_ml.explainers.integrated_gradients import (
    INTEGRATED_GRADIENTS_CAPABILITIES,
    _aggregate_attributions,
    _compute_all_attributions,
    _compute_baseline,
    _compute_integrated_gradients_single,
    _compute_interpolated_inputs,
    _validate_inputs,
)
from platform_ml.explainers.protocol import GradientModelProtocol

from .array_helpers import assert_close, get_float, make_float64_2d


class ConstantGradientModel:
    """Model with constant gradients for testing.

    Returns gradient of 1.0 for all features and samples.
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        proba[:, 0] = 0.5
        proba[:, 1] = 0.5
        return proba

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Return constant gradients of 1.0.

        Args:
            x: Input features with shape (n_samples, n_features).
            target_class: Class index for gradient computation.

        Returns:
            Gradients with shape (n_samples, n_features).
        """
        return np.ones_like(x, dtype=np.float64)


class LinearGradientModel:
    """Model with gradients that scale with feature index.

    Gradient for feature j is (j+1) * 0.1.
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        proba[:, 0] = 0.5
        proba[:, 1] = 0.5
        return proba

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Return gradients proportional to feature index.

        Args:
            x: Input features with shape (n_samples, n_features).
            target_class: Class index for gradient computation.

        Returns:
            Gradients with shape (n_samples, n_features).
        """
        n_samples = int(x.shape[0])
        n_features = int(x.shape[1])
        gradients: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)
        for j in range(n_features):
            gradients[:, j] = float(j + 1) * 0.1
        return gradients


class ModelWithoutGradients:
    """Model that only has predict_proba, no compute_gradients.

    Used to test error handling when model lacks gradient support.
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Probabilities with shape (n_samples, 2).
        """
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.zeros((n_samples, 2), dtype=np.float64)
        proba[:, 0] = 0.5
        proba[:, 1] = 0.5
        return proba


def test_integrated_gradients_capabilities_values() -> None:
    """Verify INTEGRATED_GRADIENTS_CAPABILITIES has correct values."""
    assert INTEGRATED_GRADIENTS_CAPABILITIES["requires_gradients"] is True
    assert INTEGRATED_GRADIENTS_CAPABILITIES["requires_background_data"] is True
    assert INTEGRATED_GRADIENTS_CAPABILITIES["computational_cost"] == "high"


def test_validate_inputs_matching_dimensions() -> None:
    """Verify _validate_inputs passes with matching dimensions."""
    x = make_float64_2d([[1.0, 2.0]])
    feature_names = ["a", "b"]
    _validate_inputs(x, feature_names)


def test_validate_inputs_mismatched_dimensions_raises() -> None:
    """Verify _validate_inputs raises with mismatched dimensions."""
    x = make_float64_2d([[1.0]])
    feature_names = ["a", "b"]

    with pytest.raises(ValueError, match=r"feature_names length.*must match x_data columns"):
        _validate_inputs(x, feature_names)


def test_compute_baseline_zeros_mode() -> None:
    """Verify _compute_baseline returns zeros with baseline_mode='zeros'."""
    x = make_float64_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    baseline = _compute_baseline(x, "zeros")

    assert baseline.shape == (1, 3)
    assert get_float(baseline, 0, 0) == 0.0
    assert get_float(baseline, 0, 1) == 0.0
    assert get_float(baseline, 0, 2) == 0.0


def test_compute_baseline_mean_mode() -> None:
    """Verify _compute_baseline returns mean with baseline_mode='mean'."""
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0]])

    baseline = _compute_baseline(x, "mean")

    assert baseline.shape == (1, 2)
    # Mean of [1, 3] = 2, mean of [2, 4] = 3
    assert get_float(baseline, 0, 0) == 2.0
    assert get_float(baseline, 0, 1) == 3.0


def test_compute_baseline_zeros_single_sample() -> None:
    """Verify _compute_baseline zeros with single sample."""
    x = make_float64_2d([[5.0, 6.0]])

    baseline = _compute_baseline(x, "zeros")

    assert baseline.shape == (1, 2)
    assert get_float(baseline, 0, 0) == 0.0
    assert get_float(baseline, 0, 1) == 0.0


def test_compute_baseline_mean_single_sample() -> None:
    """Verify _compute_baseline mean with single sample (mean equals input)."""
    x = make_float64_2d([[5.0, 6.0]])

    baseline = _compute_baseline(x, "mean")

    assert baseline.shape == (1, 2)
    assert get_float(baseline, 0, 0) == 5.0
    assert get_float(baseline, 0, 1) == 6.0


def test_compute_interpolated_inputs_correct_shape() -> None:
    """Verify _compute_interpolated_inputs returns correct shape."""
    x_sample = make_float64_2d([[1.0, 2.0]])
    baseline = make_float64_2d([[0.0, 0.0]])

    result = _compute_interpolated_inputs(x_sample, baseline, n_steps=10)

    assert result.shape == (10, 2)


def test_compute_interpolated_inputs_endpoints() -> None:
    """Verify _compute_interpolated_inputs has correct start and end."""
    x_sample = make_float64_2d([[4.0, 8.0]])
    baseline = make_float64_2d([[0.0, 0.0]])

    result = _compute_interpolated_inputs(x_sample, baseline, n_steps=5)

    # First step (alpha=0) should be baseline
    assert get_float(result, 0, 0) == 0.0
    assert get_float(result, 0, 1) == 0.0
    # Last step (alpha=1) should be input
    assert get_float(result, 4, 0) == 4.0
    assert get_float(result, 4, 1) == 8.0


def test_compute_interpolated_inputs_intermediate_values() -> None:
    """Verify _compute_interpolated_inputs computes correct intermediate values."""
    x_sample = make_float64_2d([[10.0]])
    baseline = make_float64_2d([[0.0]])

    result = _compute_interpolated_inputs(x_sample, baseline, n_steps=11)

    # With 11 steps, alpha goes from 0 to 1 in increments of 0.1
    assert_close(get_float(result, 0, 0), 0.0)
    assert_close(get_float(result, 5, 0), 5.0)
    assert_close(get_float(result, 10, 0), 10.0)


def test_compute_interpolated_inputs_nonzero_baseline() -> None:
    """Verify _compute_interpolated_inputs works with nonzero baseline."""
    x_sample = make_float64_2d([[6.0]])
    baseline = make_float64_2d([[2.0]])

    result = _compute_interpolated_inputs(x_sample, baseline, n_steps=5)

    # baseline + alpha * (input - baseline) = 2 + alpha * 4
    # alpha = [0, 0.25, 0.5, 0.75, 1.0]
    assert_close(get_float(result, 0, 0), 2.0)
    assert_close(get_float(result, 2, 0), 4.0)
    assert_close(get_float(result, 4, 0), 6.0)


def test_compute_integrated_gradients_single_constant_gradient() -> None:
    """Verify integrated gradients with constant gradient model.

    For constant gradient g=1, IG = (x - baseline) * 1 = x - baseline
    """
    model: GradientModelProtocol = ConstantGradientModel()
    x_sample = make_float64_2d([[4.0, 8.0]])
    baseline = make_float64_2d([[0.0, 0.0]])

    result = _compute_integrated_gradients_single(
        model=model,
        x_sample=x_sample,
        baseline=baseline,
        target_class=1,
        n_steps=50,
    )

    assert result.shape == (2,)
    # With constant gradient of 1.0, attribution = (input - baseline) * 1
    assert abs(get_float(result, 0) - 4.0) < 0.1
    assert abs(get_float(result, 1) - 8.0) < 0.1


def test_compute_integrated_gradients_single_with_nonzero_baseline() -> None:
    """Verify integrated gradients with nonzero baseline."""
    model: GradientModelProtocol = ConstantGradientModel()
    x_sample = make_float64_2d([[6.0]])
    baseline = make_float64_2d([[2.0]])

    result = _compute_integrated_gradients_single(
        model=model,
        x_sample=x_sample,
        baseline=baseline,
        target_class=1,
        n_steps=100,
    )

    assert result.shape == (1,)
    # (6 - 2) * 1 = 4
    assert abs(get_float(result, 0) - 4.0) < 0.1


def test_compute_integrated_gradients_single_zero_diff() -> None:
    """Verify integrated gradients when input equals baseline."""
    model: GradientModelProtocol = ConstantGradientModel()
    x_sample = make_float64_2d([[5.0, 5.0]])
    baseline = make_float64_2d([[5.0, 5.0]])

    result = _compute_integrated_gradients_single(
        model=model,
        x_sample=x_sample,
        baseline=baseline,
        target_class=1,
        n_steps=50,
    )

    assert result.shape == (2,)
    # (x - baseline) = 0, so attributions should be 0
    assert abs(get_float(result, 0)) < 1e-10
    assert abs(get_float(result, 1)) < 1e-10


def test_compute_all_attributions_single_sample() -> None:
    """Verify _compute_all_attributions with single sample."""
    model: GradientModelProtocol = ConstantGradientModel()
    x = make_float64_2d([[3.0, 6.0]])
    baseline = make_float64_2d([[0.0, 0.0]])

    result = _compute_all_attributions(
        model=model,
        x_data=x,
        baseline=baseline,
        target_class=1,
        n_steps=50,
    )

    assert result.shape == (1, 2)
    assert abs(get_float(result, 0, 0) - 3.0) < 0.1
    assert abs(get_float(result, 0, 1) - 6.0) < 0.1


def test_compute_all_attributions_multiple_samples() -> None:
    """Verify _compute_all_attributions with multiple samples."""
    model: GradientModelProtocol = ConstantGradientModel()
    x = make_float64_2d([[2.0, 4.0], [1.0, 3.0]])
    baseline = make_float64_2d([[0.0, 0.0]])

    result = _compute_all_attributions(
        model=model,
        x_data=x,
        baseline=baseline,
        target_class=1,
        n_steps=50,
    )

    assert result.shape == (2, 2)
    # First sample
    assert abs(get_float(result, 0, 0) - 2.0) < 0.1
    assert abs(get_float(result, 0, 1) - 4.0) < 0.1
    # Second sample
    assert abs(get_float(result, 1, 0) - 1.0) < 0.1
    assert abs(get_float(result, 1, 1) - 3.0) < 0.1


def test_aggregate_attributions_single_sample() -> None:
    """Verify _aggregate_attributions with single sample."""
    attr = make_float64_2d([[0.5, -0.3]])

    result = _aggregate_attributions(attr)

    assert result.shape == (2,)
    # Mean absolute: [0.5, 0.3]
    assert_close(get_float(result, 0), 0.5)
    assert_close(get_float(result, 1), 0.3)


def test_aggregate_attributions_multiple_samples() -> None:
    """Verify _aggregate_attributions averages correctly."""
    attr = make_float64_2d([[0.2, 0.4], [0.6, 0.8]])

    result = _aggregate_attributions(attr)

    assert result.shape == (2,)
    # Mean of [0.2, 0.6] = 0.4, mean of [0.4, 0.8] = 0.6
    assert_close(get_float(result, 0), 0.4)
    assert_close(get_float(result, 1), 0.6)
