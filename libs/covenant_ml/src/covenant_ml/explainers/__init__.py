"""Explainer integration for covenant_ml backends.

Provides a registry of feature importance explainers with explicit backend
compatibility. Each explainer declares which backends it supports:

- permutation: All backends (xgboost, lightgbm, mlp, lstm)
- gradient: Neural networks only (mlp, lstm) - requires compute_gradients()
- integrated_gradients: Neural networks only (mlp, lstm) - requires compute_gradients()
- shap_tree: Tree models only (xgboost, lightgbm)

Usage:
    from covenant_ml.explainers import ExplainerName, default_explainer_registry

    # Get registry
    registry = default_explainer_registry()

    # Check compatibility
    if registry.is_compatible(ExplainerName.GRADIENT, "mlp"):
        explainer = registry.get(ExplainerName.GRADIENT)
        importance = explainer.compute_importance(
            model=model,
            x_data=x_test,
            feature_names=feature_names,
            target_class=1,
        )

    # List compatible explainers for a backend
    compatible = registry.list_compatible_explainers("xgboost")
    # Returns: [ExplainerName.PERMUTATION, ExplainerName.SHAP_TREE]
"""

from __future__ import annotations

from .registry import (
    ExplainerFactory,
    ExplainerRegistration,
    ExplainerRegistry,
    default_explainer_registry,
)
from .regression_registry import (
    RegressionExplainerFactory,
    RegressionExplainerRegistration,
    RegressionExplainerRegistry,
    default_regression_explainer_registry,
)
from .types import (
    ComputationalCost,
    ExplainerCapabilities,
    ExplainerName,
    ExplainResult,
    FeatureImportanceScore,
    GradientConfig,
    IntegratedGradientsConfig,
    PermutationConfig,
    RegressionExplainResult,
)

__all__ = [
    "ComputationalCost",
    "ExplainResult",
    "ExplainerCapabilities",
    "ExplainerFactory",
    "ExplainerName",
    "ExplainerRegistration",
    "ExplainerRegistry",
    "FeatureImportanceScore",
    "GradientConfig",
    "IntegratedGradientsConfig",
    "PermutationConfig",
    "RegressionExplainResult",
    "RegressionExplainerFactory",
    "RegressionExplainerRegistration",
    "RegressionExplainerRegistry",
    "default_explainer_registry",
    "default_regression_explainer_registry",
]
