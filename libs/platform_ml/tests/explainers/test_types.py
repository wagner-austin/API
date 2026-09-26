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


def test_explainer_names_are_the_registry_words() -> None:
    """Each explainer's name is the word the explainer registries key it by."""
    assert [name.value for name in ExplainerName] == [
        "permutation",
        "gradient",
        "integrated_gradients",
        "shap_tree",
    ]


def test_computational_costs_are_ordered_cheapest_first() -> None:
    """The cost categories and their wire words, cheapest first."""
    assert [cost.value for cost in ComputationalCost] == ["low", "medium", "high"]


def test_explainer_capabilities_creation() -> None:
    """Verify ExplainerCapabilities TypedDict can be instantiated."""
    caps: ExplainerCapabilities = {
        "requires_gradients": True,
        "requires_background_data": False,
        "computational_cost": ComputationalCost.HIGH,
    }
    assert caps["requires_gradients"] is True
    assert caps["requires_background_data"] is False
    assert caps["computational_cost"] is ComputationalCost.HIGH


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
