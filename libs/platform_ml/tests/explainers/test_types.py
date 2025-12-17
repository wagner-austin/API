"""Tests for platform_ml.explainers.types module.

Achieves 100% statement and branch coverage by testing all TypedDict
instantiation and Literal type usage patterns.
"""

from __future__ import annotations

from platform_ml.explainers.types import (
    ComputationalCost,
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
    GradientConfig,
    IntegratedGradientsConfig,
    PermutationConfig,
)


def test_explainer_name_permutation() -> None:
    """Verify ExplainerName accepts 'permutation'."""
    name: ExplainerName = "permutation"
    assert name == "permutation"


def test_explainer_name_gradient() -> None:
    """Verify ExplainerName accepts 'gradient'."""
    name: ExplainerName = "gradient"
    assert name == "gradient"


def test_explainer_name_integrated_gradients() -> None:
    """Verify ExplainerName accepts 'integrated_gradients'."""
    name: ExplainerName = "integrated_gradients"
    assert name == "integrated_gradients"


def test_computational_cost_low() -> None:
    """Verify ComputationalCost accepts 'low'."""
    cost: ComputationalCost = "low"
    assert cost == "low"


def test_computational_cost_medium() -> None:
    """Verify ComputationalCost accepts 'medium'."""
    cost: ComputationalCost = "medium"
    assert cost == "medium"


def test_computational_cost_high() -> None:
    """Verify ComputationalCost accepts 'high'."""
    cost: ComputationalCost = "high"
    assert cost == "high"


def test_explainer_capabilities_creation() -> None:
    """Verify ExplainerCapabilities TypedDict can be instantiated."""
    caps: ExplainerCapabilities = {
        "requires_gradients": True,
        "requires_background_data": False,
        "computational_cost": "high",
    }
    assert caps["requires_gradients"] is True
    assert caps["requires_background_data"] is False
    assert caps["computational_cost"] == "high"


def test_feature_importance_score_creation() -> None:
    """Verify FeatureImportanceScore TypedDict can be instantiated."""
    score: FeatureImportanceScore = {
        "name": "feature_a",
        "importance": 0.75,
        "rank": 1,
    }
    assert score["name"] == "feature_a"
    assert score["importance"] == 0.75
    assert score["rank"] == 1


def test_permutation_config_creation() -> None:
    """Verify PermutationConfig TypedDict can be instantiated."""
    config: PermutationConfig = {
        "n_repeats": 10,
        "random_state": 42,
    }
    assert config["n_repeats"] == 10
    assert config["random_state"] == 42


def test_gradient_config_creation() -> None:
    """Verify GradientConfig TypedDict can be instantiated."""
    config: GradientConfig = {
        "multiply_by_input": True,
        "absolute_value": False,
    }
    assert config["multiply_by_input"] is True
    assert config["absolute_value"] is False


def test_integrated_gradients_config_zeros_baseline() -> None:
    """Verify IntegratedGradientsConfig with zeros baseline."""
    config: IntegratedGradientsConfig = {
        "n_steps": 50,
        "baseline_mode": "zeros",
    }
    assert config["n_steps"] == 50
    assert config["baseline_mode"] == "zeros"


def test_integrated_gradients_config_mean_baseline() -> None:
    """Verify IntegratedGradientsConfig with mean baseline."""
    config: IntegratedGradientsConfig = {
        "n_steps": 100,
        "baseline_mode": "mean",
    }
    assert config["n_steps"] == 100
    assert config["baseline_mode"] == "mean"
