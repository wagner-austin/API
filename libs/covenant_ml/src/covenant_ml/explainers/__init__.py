"""Explainer integration for covenant_ml backends.

Provides a registry of feature importance explainers with explicit backend
compatibility. Each explainer declares which backends it supports:

- permutation: All backends (xgboost, lightgbm, mlp, lstm)
- gradient: Neural networks only (mlp, lstm) - requires compute_gradients()
- integrated_gradients: Neural networks only (mlp, lstm) - requires compute_gradients()
- shap_tree: Tree models only (xgboost, lightgbm)

Usage:
    from covenant_ml.explainers import default_explainer_registry, ExplainResult

    # Get registry
    registry = default_explainer_registry()

    # Check compatibility
    if registry.is_compatible("gradient", "mlp"):
        explainer = registry.get("gradient")
        importance = explainer.compute_importance(
            model=model,
            x_data=x_test,
            feature_names=feature_names,
            target_class=1,
        )

    # List compatible explainers for a backend
    compatible = registry.list_compatible_explainers("xgboost")
    # Returns: ["permutation", "shap_tree"]
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
    ExplainerConfigUnion,
    ExplainerName,
    ExplainRequestConfig,
    ExplainResult,
    FeatureImportanceScore,
    GradientConfig,
    GradientExplainConfig,
    IntegratedGradientsConfig,
    IntegratedGradientsExplainConfig,
    PermutationConfig,
    PermutationExplainConfig,
    RegressionExplainResult,
    ShapTreeExplainConfig,
    SupportedExplainer,
)

__all__ = [
    "ComputationalCost",
    "ExplainRequestConfig",
    "ExplainResult",
    "ExplainerCapabilities",
    "ExplainerConfigUnion",
    "ExplainerFactory",
    "ExplainerName",
    "ExplainerRegistration",
    "ExplainerRegistry",
    "FeatureImportanceScore",
    "GradientConfig",
    "GradientExplainConfig",
    "IntegratedGradientsConfig",
    "IntegratedGradientsExplainConfig",
    "PermutationConfig",
    "PermutationExplainConfig",
    "RegressionExplainResult",
    "RegressionExplainerFactory",
    "RegressionExplainerRegistration",
    "RegressionExplainerRegistry",
    "ShapTreeExplainConfig",
    "SupportedExplainer",
    "default_explainer_registry",
    "default_regression_explainer_registry",
]
