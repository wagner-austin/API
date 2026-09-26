"""Types for explainer integration with covenant_ml backends.

Provides the result TypedDicts for feature importance computation.
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
    explainer: ExplainerName
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
    explainer: ExplainerName
    n_samples_used: int
    n_features: int
    feature_importances: list[FeatureImportanceScore]
    duration_seconds: float


__all__ = [
    "ComputationalCost",
    "ExplainResult",
    "ExplainerCapabilities",
    "ExplainerName",
    "FeatureImportanceScore",
    "GradientConfig",
    "IntegratedGradientsConfig",
    "PermutationConfig",
    "RegressionExplainResult",
]
