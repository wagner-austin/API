"""Types and TypedDicts for feature importance explainers.

Defines strict types for explainer configurations, capabilities, and results.
No Any, cast, or type: ignore allowed.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

ExplainerName = Literal["permutation", "gradient", "integrated_gradients", "shap_tree"]
"""Supported explainer names."""

ComputationalCost = Literal["low", "medium", "high"]
"""Computational cost category for an explainer."""


class ExplainerCapabilities(TypedDict, total=True):
    """Describes supported features of an explainer implementation.

    Args:
        requires_gradients: True if the explainer needs gradient computation.
            Only works with differentiable models (MLP, LSTM).
        requires_background_data: True if the explainer needs background/baseline data.
        computational_cost: Relative computational cost category.
    """

    requires_gradients: bool
    requires_background_data: bool
    computational_cost: ComputationalCost


class FeatureImportanceScore(TypedDict, total=True):
    """Single feature importance result.

    Args:
        name: Feature name.
        importance: Importance score (higher = more important).
        rank: Rank among all features (1 = most important).
    """

    name: str
    importance: float
    rank: int


class PermutationConfig(TypedDict, total=True):
    """Configuration for permutation importance explainer.

    Args:
        n_repeats: Number of times to shuffle each feature.
        random_state: Random seed for reproducibility.
    """

    n_repeats: int
    random_state: int


class GradientConfig(TypedDict, total=True):
    """Configuration for gradient-based explainer.

    Args:
        multiply_by_input: Whether to multiply gradients by input values.
        absolute_value: Whether to use absolute values of attributions.
    """

    multiply_by_input: bool
    absolute_value: bool


class IntegratedGradientsConfig(TypedDict, total=True):
    """Configuration for integrated gradients explainer.

    Args:
        n_steps: Number of interpolation steps from baseline to input.
        baseline_mode: How to compute baseline ("zeros" or "mean").
    """

    n_steps: int
    baseline_mode: Literal["zeros", "mean"]


__all__ = [
    "ComputationalCost",
    "ExplainerCapabilities",
    "ExplainerName",
    "FeatureImportanceScore",
    "GradientConfig",
    "IntegratedGradientsConfig",
    "PermutationConfig",
]
