"""Tests for integrated gradients: test_integrated_gradients_explainer_init_zeros."""

from __future__ import annotations

import pytest

from platform_ml.explainers.integrated_gradients import (
    IntegratedGradientsExplainer,
    _GradientModelWrapper,
    create_integrated_gradients_explainer,
)
from platform_ml.explainers.protocol import GradientModelProtocol
from platform_ml.explainers.types import IntegratedGradientsConfig
from tests.explainers.test_integrated_gradients_internals import (
    ConstantGradientModel,
    LinearGradientModel,
    ModelWithoutGradients,
)

from .array_helpers import assert_close, get_float, make_float64_2d


def test_integrated_gradients_explainer_init_zeros() -> None:
    """Verify IntegratedGradientsExplainer initializes with zeros baseline."""
    config: IntegratedGradientsConfig = {"n_steps": 50, "baseline_mode": "zeros"}
    explainer = IntegratedGradientsExplainer(config)

    assert explainer._n_steps == 50
    assert explainer._baseline_mode == "zeros"


def test_integrated_gradients_explainer_init_mean() -> None:
    """Verify IntegratedGradientsExplainer initializes with mean baseline."""
    config: IntegratedGradientsConfig = {"n_steps": 100, "baseline_mode": "mean"}
    explainer = IntegratedGradientsExplainer(config)

    assert explainer._n_steps == 100
    assert explainer._baseline_mode == "mean"


def test_integrated_gradients_explainer_name() -> None:
    """Verify IntegratedGradientsExplainer.explainer_name returns correct value."""
    config: IntegratedGradientsConfig = {"n_steps": 50, "baseline_mode": "zeros"}
    explainer = IntegratedGradientsExplainer(config)

    assert explainer.explainer_name() == "integrated_gradients"


def test_integrated_gradients_explainer_capabilities() -> None:
    """Verify IntegratedGradientsExplainer.capabilities returns correct values."""
    config: IntegratedGradientsConfig = {"n_steps": 50, "baseline_mode": "zeros"}
    explainer = IntegratedGradientsExplainer(config)

    caps = explainer.capabilities()

    assert caps["requires_gradients"] is True
    assert caps["requires_background_data"] is True
    assert caps["computational_cost"] == "high"


def test_integrated_gradients_explainer_validates_input() -> None:
    """Verify compute_importance validates input dimensions."""
    config: IntegratedGradientsConfig = {"n_steps": 10, "baseline_mode": "zeros"}
    explainer = IntegratedGradientsExplainer(config)
    model: GradientModelProtocol = ConstantGradientModel()
    x = make_float64_2d([[1.0, 2.0, 3.0]])
    feature_names = ["a"]

    with pytest.raises(ValueError, match=r"feature_names length"):
        explainer.compute_importance(
            model=model,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )


def test_integrated_gradients_explainer_compute_importance_zeros() -> None:
    """Verify compute_importance with zeros baseline."""
    config: IntegratedGradientsConfig = {"n_steps": 50, "baseline_mode": "zeros"}
    explainer = IntegratedGradientsExplainer(config)
    model: GradientModelProtocol = ConstantGradientModel()
    x = make_float64_2d([[2.0, 6.0]])
    feature_names = ["small", "large"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(result) == 2
    # With constant gradient=1 and zeros baseline, importance = abs(input)
    # So "large" (6.0) should rank higher than "small" (2.0)
    assert result[0]["name"] == "large"
    assert result[0]["rank"] == 1
    assert result[1]["name"] == "small"
    assert result[1]["rank"] == 2


def test_integrated_gradients_explainer_compute_importance_mean() -> None:
    """Verify compute_importance with mean baseline."""
    config: IntegratedGradientsConfig = {"n_steps": 50, "baseline_mode": "mean"}
    explainer = IntegratedGradientsExplainer(config)
    model: GradientModelProtocol = ConstantGradientModel()
    # Create data where mean baseline leads to different relative importances
    x = make_float64_2d([[2.0, 8.0], [4.0, 4.0]])
    feature_names = ["feat_a", "feat_b"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(result) == 2
    # Mean baseline: [3.0, 6.0]
    # Sample 1 diff: [-1.0, 2.0], Sample 2 diff: [1.0, -2.0]
    # Attributions (constant grad=1): same as diff
    # Mean abs: [1.0, 2.0]
    assert result[0]["name"] == "feat_b"


def test_integrated_gradients_explainer_linear_model() -> None:
    """Verify compute_importance with linear gradient model."""
    config: IntegratedGradientsConfig = {"n_steps": 100, "baseline_mode": "zeros"}
    explainer = IntegratedGradientsExplainer(config)
    model: GradientModelProtocol = LinearGradientModel()
    x = make_float64_2d([[1.0, 1.0, 1.0]])
    feature_names = ["first", "second", "third"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=1,
    )

    assert len(result) == 3
    # Linear model has gradient [0.1, 0.2, 0.3]
    # All inputs are 1.0, baseline is 0.0
    # Attributions should be proportional to gradients
    assert result[0]["name"] == "third"
    assert result[1]["name"] == "second"
    assert result[2]["name"] == "first"


def test_create_integrated_gradients_explainer_returns_configured_instance() -> None:
    """Verify create_integrated_gradients_explainer returns properly configured instance."""
    config: IntegratedGradientsConfig = {"n_steps": 25, "baseline_mode": "mean"}

    explainer = create_integrated_gradients_explainer(config)

    assert explainer._n_steps == 25
    assert explainer._baseline_mode == "mean"
    assert explainer.explainer_name() == "integrated_gradients"
    assert explainer.capabilities()["requires_gradients"] is True


def test_create_integrated_gradients_explainer_functional() -> None:
    """Verify factory-created explainer works correctly."""
    config: IntegratedGradientsConfig = {"n_steps": 50, "baseline_mode": "zeros"}
    explainer = create_integrated_gradients_explainer(config)
    model: GradientModelProtocol = ConstantGradientModel()
    x = make_float64_2d([[3.0, 1.0]])
    feature_names = ["x", "y"]

    result = explainer.compute_importance(
        model=model,
        x_data=x,
        feature_names=feature_names,
        target_class=0,
    )

    assert len(result) == 2
    # With constant gradient and zeros baseline, x (3.0) should rank higher
    assert result[0]["name"] == "x"


def test_gradient_model_wrapper_predict_proba() -> None:
    """Verify _GradientModelWrapper.predict_proba delegates to wrapped model."""
    model = ConstantGradientModel()
    wrapper = _GradientModelWrapper(model)

    x = make_float64_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    proba = wrapper.predict_proba(x)

    # Model returns uniform 0.5 probabilities
    assert proba.shape == (2, 2)
    assert_close(get_float(proba, 0, 0), 0.5)
    assert_close(get_float(proba, 0, 1), 0.5)
    assert_close(get_float(proba, 1, 0), 0.5)
    assert_close(get_float(proba, 1, 1), 0.5)


def test_compute_importance_raises_without_gradients() -> None:
    """Verify compute_importance raises AttributeError for model without compute_gradients."""
    config: IntegratedGradientsConfig = {"n_steps": 10, "baseline_mode": "zeros"}
    explainer = IntegratedGradientsExplainer(config)
    model = ModelWithoutGradients()
    x = make_float64_2d([[1.0, 2.0]])
    feature_names = ["a", "b"]

    with pytest.raises(AttributeError, match=r"must have compute_gradients"):
        explainer.compute_importance(
            model=model,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )
