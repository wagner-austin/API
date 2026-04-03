"""Types for explainer integration with covenant_ml backends.

Provides configuration and result TypedDicts for feature importance computation.
Re-exports relevant types from platform_ml.explainers for convenience.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_ml.explainers.types import (
    ComputationalCost,
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
    GradientConfig,
    IntegratedGradientsConfig,
    PermutationConfig,
)

# Explainer name literal including ShapTree wrapper
SupportedExplainer = Literal["permutation", "gradient", "integrated_gradients", "shap_tree"]


class ExplainRequestConfig(TypedDict, total=True):
    """Configuration for an explanation request.

    Args:
        explainer: Which explainer to use.
        target_class: Class index for importance computation (typically 1 for binary).
        n_samples: Number of samples to use for explanation (subset for speed).
        random_state: Random seed for reproducibility.
    """

    explainer: SupportedExplainer
    target_class: int
    n_samples: int
    random_state: int


class PermutationExplainConfig(TypedDict, total=True):
    """Extended config for permutation explainer requests.

    Args:
        explainer: Must be "permutation".
        target_class: Class index for importance computation.
        n_samples: Number of samples to use for explanation.
        random_state: Random seed for reproducibility.
        n_repeats: Number of times to shuffle each feature.
    """

    explainer: Literal["permutation"]
    target_class: int
    n_samples: int
    random_state: int
    n_repeats: int


class GradientExplainConfig(TypedDict, total=True):
    """Extended config for gradient explainer requests.

    Args:
        explainer: Must be "gradient".
        target_class: Class index for importance computation.
        n_samples: Number of samples to use for explanation.
        random_state: Random seed for reproducibility.
        multiply_by_input: Whether to multiply gradients by input values.
        absolute_value: Whether to use absolute values of attributions.
    """

    explainer: Literal["gradient"]
    target_class: int
    n_samples: int
    random_state: int
    multiply_by_input: bool
    absolute_value: bool


class IntegratedGradientsExplainConfig(TypedDict, total=True):
    """Extended config for integrated gradients requests.

    Args:
        explainer: Must be "integrated_gradients".
        target_class: Class index for importance computation.
        n_samples: Number of samples to use for explanation.
        random_state: Random seed for reproducibility.
        n_steps: Number of interpolation steps.
        baseline_mode: How to compute baseline ("zeros" or "mean").
    """

    explainer: Literal["integrated_gradients"]
    target_class: int
    n_samples: int
    random_state: int
    n_steps: int
    baseline_mode: Literal["zeros", "mean"]


class ShapTreeExplainConfig(TypedDict, total=True):
    """Extended config for SHAP tree explainer requests.

    Args:
        explainer: Must be "shap_tree".
        target_class: Class index for importance computation.
        n_samples: Number of samples to use for explanation.
        random_state: Random seed for reproducibility.
    """

    explainer: Literal["shap_tree"]
    target_class: int
    n_samples: int
    random_state: int


# Union of all explainer-specific configs
ExplainerConfigUnion = (
    PermutationExplainConfig
    | GradientExplainConfig
    | IntegratedGradientsExplainConfig
    | ShapTreeExplainConfig
)


class ExplainResult(TypedDict, total=True):
    """Result of feature importance computation.

    Args:
        status: Completion status ("complete" or "failed").
        backend: Backend that was explained.
        explainer: Explainer used.
        n_samples_used: Actual number of samples used.
        n_features: Number of features in the model.
        target_class: Class index that was explained.
        feature_importances: Ranked list of feature importance scores.
        duration_seconds: Time taken for computation.
    """

    status: Literal["complete", "failed"]
    backend: str
    explainer: SupportedExplainer
    n_samples_used: int
    n_features: int
    target_class: int
    feature_importances: list[FeatureImportanceScore]
    duration_seconds: float


class RegressionExplainResult(TypedDict, total=True):
    """Result of regression feature importance computation.

    Like ExplainResult but without target_class (regression has single output).

    Args:
        status: Completion status ("complete" or "failed").
        backend: Regressor backend that was explained.
        explainer: Explainer used.
        n_samples_used: Actual number of samples used.
        n_features: Number of features in the model.
        feature_importances: Ranked list of feature importance scores.
        duration_seconds: Time taken for computation.
    """

    status: Literal["complete", "failed"]
    backend: str
    explainer: SupportedExplainer
    n_samples_used: int
    n_features: int
    feature_importances: list[FeatureImportanceScore]
    duration_seconds: float


__all__ = [
    "ComputationalCost",
    "ExplainRequestConfig",
    "ExplainResult",
    "ExplainerCapabilities",
    "ExplainerConfigUnion",
    "ExplainerName",
    "FeatureImportanceScore",
    "GradientConfig",
    "GradientExplainConfig",
    "IntegratedGradientsConfig",
    "IntegratedGradientsExplainConfig",
    "PermutationConfig",
    "PermutationExplainConfig",
    "RegressionExplainResult",
    "ShapTreeExplainConfig",
    "SupportedExplainer",
]
