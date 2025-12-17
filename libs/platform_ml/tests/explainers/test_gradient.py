"""Tests for platform_ml.explainers.gradient module.

Achieves 100% statement and branch coverage by testing all functions,
class methods, validation paths, and configuration options.
Uses real models implementing GradientModelProtocol without mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from platform_ml.explainers.gradient import (
    GRADIENT_CAPABILITIES,
    GradientExplainer,
    _aggregate_attributions,
    _compute_attributions,
    _get_importance_from_pair,
    _rank_importances,
    _validate_inputs,
    create_gradient_explainer,
)
from platform_ml.explainers.protocol import GradientModelProtocol
from platform_ml.explainers.types import GradientConfig

from .array_helpers import assert_close, get_float, make_float64_1d, make_float64_2d


class LinearGradientModel:
    """Model with linear gradients for testing.

    Returns gradients proportional to feature index.
    Feature 0 has smallest gradient, last feature has largest.
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


class SignedGradientModel:
    """Model with signed gradients for testing absolute value option.

    Returns alternating positive and negative gradients.
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
        """Return alternating positive and negative gradients.

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
            sign: float = 1.0 if j % 2 == 0 else -1.0
            gradients[:, j] = sign * 0.5
        return gradients


class ZeroGradientModel:
    """Model that returns zero gradients."""

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
        """Return zero gradients.

        Args:
            x: Input features with shape (n_samples, n_features).
            target_class: Class index for gradient computation.

        Returns:
            Zero gradients with shape (n_samples, n_features).
        """
        return np.zeros_like(x, dtype=np.float64)


def test_gradient_capabilities_values() -> None:
    """Verify GRADIENT_CAPABILITIES has correct values."""
    assert GRADIENT_CAPABILITIES["requires_gradients"] is True
    assert GRADIENT_CAPABILITIES["requires_background_data"] is False
    assert GRADIENT_CAPABILITIES["computational_cost"] == "low"


def test_validate_inputs_matching_dimensions() -> None:
    """Verify _validate_inputs passes with matching dimensions."""
    x = make_float64_2d([[1.0, 2.0]])
    feature_names = ["a", "b"]
    _validate_inputs(x, feature_names)


def test_validate_inputs_mismatched_dimensions_raises() -> None:
    """Verify _validate_inputs raises with mismatched dimensions."""
    x = make_float64_2d([[1.0, 2.0]])
    feature_names = ["a", "b", "c"]

    with pytest.raises(ValueError, match=r"feature_names length.*must match x_data columns"):
        _validate_inputs(x, feature_names)


def test_compute_attributions_no_multiply_no_abs() -> None:
    """Verify _compute_attributions without multiply_by_input or absolute_value."""
    model: GradientModelProtocol = LinearGradientModel()
    x = make_float64_2d([[1.0, 2.0, 3.0]])

    attr = _compute_attributions(
        model=model,
        x_data=x,
        target_class=1,
        multiply_by_input=False,
        absolute_value=False,
    )

    assert attr.shape == (1, 3)
    assert_close(get_float(attr, 0, 0), 0.1)
    assert_close(get_float(attr, 0, 1), 0.2)
    assert_close(get_float(attr, 0, 2), 0.3)


def test_compute_attributions_with_multiply_by_input() -> None:
    """Verify _compute_attributions with multiply_by_input=True."""
    model: GradientModelProtocol = LinearGradientModel()
    x = make_float64_2d([[2.0, 3.0]])

    attr = _compute_attributions(
        model=model,
        x_data=x,
        target_class=1,
        multiply_by_input=True,
        absolute_value=False,
    )

    assert attr.shape == (1, 2)
    # Gradients [0.1, 0.2] * inputs [2.0, 3.0] = [0.2, 0.6]
    assert_close(get_float(attr, 0, 0), 0.2)
    assert_close(get_float(attr, 0, 1), 0.6)


def test_compute_attributions_with_absolute_value() -> None:
    """Verify _compute_attributions with absolute_value=True."""
    model: GradientModelProtocol = SignedGradientModel()
    x = make_float64_2d([[1.0, 1.0, 1.0]])

    attr = _compute_attributions(
        model=model,
        x_data=x,
        target_class=1,
        multiply_by_input=False,
        absolute_value=True,
    )

    assert attr.shape == (1, 3)
    # All should be positive 0.5 after absolute value
    assert get_float(attr, 0, 0) == 0.5
    assert get_float(attr, 0, 1) == 0.5
    assert get_float(attr, 0, 2) == 0.5


def test_compute_attributions_multiply_and_absolute() -> None:
    """Verify _compute_attributions with both options enabled."""
    model: GradientModelProtocol = SignedGradientModel()
    x = make_float64_2d([[2.0, -3.0]])

    attr = _compute_attributions(
        model=model,
        x_data=x,
        target_class=1,
        multiply_by_input=True,
        absolute_value=True,
    )

    assert attr.shape == (1, 2)
    # Gradients [0.5, -0.5] * inputs [2.0, -3.0] = [1.0, 1.5]
    assert_close(get_float(attr, 0, 0), 1.0)
    assert_close(get_float(attr, 0, 1), 1.5)


def test_compute_attributions_multiple_samples() -> None:
    """Verify _compute_attributions works with multiple samples."""
    model: GradientModelProtocol = LinearGradientModel()
    x = make_float64_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    attr = _compute_attributions(
        model=model,
        x_data=x,
        target_class=0,
        multiply_by_input=False,
        absolute_value=False,
    )

    assert attr.shape == (3, 2)


def test_aggregate_attributions_single_sample() -> None:
    """Verify _aggregate_attributions with single sample."""
    attr = make_float64_2d([[0.1, 0.2, 0.3]])

    result = _aggregate_attributions(attr)

    assert result.shape == (3,)
    assert_close(get_float(result, 0), 0.1)
    assert_close(get_float(result, 1), 0.2)
    assert_close(get_float(result, 2), 0.3)


def test_aggregate_attributions_multiple_samples() -> None:
    """Verify _aggregate_attributions averages across samples."""
    attr = make_float64_2d([[0.2, 0.4], [0.4, 0.6]])

    result = _aggregate_attributions(attr)

    assert result.shape == (2,)
    # Mean of [0.2, 0.4] and [0.4, 0.6]
    assert_close(get_float(result, 0), 0.3)
    assert_close(get_float(result, 1), 0.5)


def test_aggregate_attributions_handles_negatives() -> None:
    """Verify _aggregate_attributions takes absolute values."""
    attr = make_float64_2d([[-0.2, 0.4], [0.2, -0.6]])

    result = _aggregate_attributions(attr)

    assert result.shape == (2,)
    # abs: [[0.2, 0.4], [0.2, 0.6]], mean: [0.2, 0.5]
    assert_close(get_float(result, 0), 0.2)
    assert_close(get_float(result, 1), 0.5)


def test_get_importance_from_pair_int_index() -> None:
    """Verify _get_importance_from_pair extracts importance from int-indexed pair."""
    pair: tuple[int, float] = (5, 0.42)
    result = _get_importance_from_pair(pair)
    assert result == 0.42


def test_get_importance_from_pair_zero() -> None:
    """Verify _get_importance_from_pair works with zero importance."""
    pair: tuple[int, float] = (0, 0.0)
    result = _get_importance_from_pair(pair)
    assert result == 0.0


def test_rank_importances_correct_ordering() -> None:
    """Verify _rank_importances sorts correctly."""
    feature_names = ["low", "high", "medium"]
    importances = make_float64_1d([0.1, 0.9, 0.5])

    result = _rank_importances(feature_names, importances)

    assert len(result) == 3
    assert result[0]["name"] == "high"
    assert result[0]["rank"] == 1
    assert result[1]["name"] == "medium"
    assert result[1]["rank"] == 2
    assert result[2]["name"] == "low"
    assert result[2]["rank"] == 3


def test_rank_importances_uses_item_for_extraction() -> None:
    """Verify _rank_importances correctly extracts values using .item()."""
    feature_names = ["a", "b"]
    importances = make_float64_1d([0.25, 0.75])

    result = _rank_importances(feature_names, importances)

    assert result[0]["importance"] == 0.75
    assert result[1]["importance"] == 0.25


def test_gradient_explainer_init() -> None:
    """Verify GradientExplainer initializes correctly."""
    config: GradientConfig = {"multiply_by_input": True, "absolute_value": False}
    explainer = GradientExplainer(config)

    assert explainer._multiply_by_input is True
    assert explainer._absolute_value is False


def test_gradient_explainer_init_false_flags() -> None:
    """Verify GradientExplainer initializes with False flags."""
    config: GradientConfig = {"multiply_by_input": False, "absolute_value": False}
    explainer = GradientExplainer(config)

    assert explainer._multiply_by_input is False
    assert explainer._absolute_value is False


def test_gradient_explainer_name() -> None:
    """Verify GradientExplainer.explainer_name returns 'gradient'."""
    config: GradientConfig = {"multiply_by_input": False, "absolute_value": True}
    explainer = GradientExplainer(config)

    assert explainer.explainer_name() == "gradient"


def test_gradient_explainer_capabilities() -> None:
    """Verify GradientExplainer.capabilities returns correct values."""
    config: GradientConfig = {"multiply_by_input": False, "absolute_value": True}
    explainer = GradientExplainer(config)

    caps = explainer.capabilities()

    assert caps["requires_gradients"] is True
    assert caps["requires_background_data"] is False
    assert caps["computational_cost"] == "low"


def test_gradient_explainer_compute_importance_validates_input() -> None:
    """Verify compute_importance validates input dimensions."""
    config: GradientConfig = {"multiply_by_input": False, "absolute_value": True}
    explainer = GradientExplainer(config)
    model: GradientModelProtocol = LinearGradientModel()
    x = make_float64_2d([[1.0, 2.0]])
    feature_names = ["a", "b", "c", "d"]

    with pytest.raises(ValueError, match=r"feature_names length"):
        explainer.compute_importance(
            model=model,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )


def test_gradient_explainer_compute_importance_basic() -> None:
    """Verify compute_importance returns correct results."""
    config: GradientConfig = {"multiply_by_input": False, "absolute_value": True}
    explainer = GradientExplainer(config)
    model: GradientModelProtocol = LinearGradientModel()
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    feature_names = ["first", "second", "third"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(result) == 3
    # Gradients are [0.1, 0.2, 0.3], so "third" should be most important
    assert result[0]["name"] == "third"
    assert result[0]["rank"] == 1
    assert result[1]["name"] == "second"
    assert result[2]["name"] == "first"


def test_gradient_explainer_compute_importance_with_multiply() -> None:
    """Verify compute_importance with multiply_by_input=True."""
    config: GradientConfig = {"multiply_by_input": True, "absolute_value": True}
    explainer = GradientExplainer(config)
    model: GradientModelProtocol = LinearGradientModel()
    # Input values affect importance when multiply_by_input=True
    x = make_float64_2d([[10.0, 1.0]])
    feature_names = ["high_input", "low_input"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(result) == 2
    # Gradients [0.1, 0.2], inputs [10.0, 1.0]
    # Attributions: [1.0, 0.2]
    assert result[0]["name"] == "high_input"
    assert result[0]["rank"] == 1


def test_gradient_explainer_zero_gradients() -> None:
    """Verify compute_importance handles zero gradients."""
    config: GradientConfig = {"multiply_by_input": False, "absolute_value": True}
    explainer = GradientExplainer(config)
    model: GradientModelProtocol = ZeroGradientModel()
    x = make_float64_2d([[1.0, 2.0]])
    feature_names = ["a", "b"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(result) == 2
    assert result[0]["importance"] == 0.0
    assert result[1]["importance"] == 0.0


def test_create_gradient_explainer_returns_configured_instance() -> None:
    """Verify create_gradient_explainer returns properly configured instance."""
    config: GradientConfig = {"multiply_by_input": True, "absolute_value": True}

    explainer = create_gradient_explainer(config)

    assert explainer._multiply_by_input is True
    assert explainer._absolute_value is True
    assert explainer.explainer_name() == "gradient"
    assert explainer.capabilities()["requires_gradients"] is True


def test_create_gradient_explainer_functional() -> None:
    """Verify factory-created explainer works correctly."""
    config: GradientConfig = {"multiply_by_input": False, "absolute_value": False}
    explainer = create_gradient_explainer(config)
    model: GradientModelProtocol = LinearGradientModel()
    x = make_float64_2d([[1.0, 2.0]])
    feature_names = ["x", "y"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=0,
    )

    assert len(result) == 2
