"""Feature importance explainers for ML models.

Provides pluggable explainers for computing feature importance:

- PermutationExplainer: Model-agnostic, works with any predictor
- GradientExplainer: Fast gradient-based, requires differentiable models
- IntegratedGradientsExplainer: Accurate path-integrated gradients

Usage:
    from platform_ml.explainers import (
        PermutationExplainer,
        PermutationConfig,
        create_permutation_explainer,
    )

    config: PermutationConfig = {"n_repeats": 10, "random_state": 42}
    explainer = create_permutation_explainer(config)
    importance = explainer.compute_importance(
        model=model,
        x_data=x_test,
        feature_names=feature_names,
        target_class=1,
    )
"""

from __future__ import annotations

from .gradient import (
    GRADIENT_CAPABILITIES,
    GradientExplainer,
    create_gradient_explainer,
)
from .integrated_gradients import (
    INTEGRATED_GRADIENTS_CAPABILITIES,
    IntegratedGradientsExplainer,
    create_integrated_gradients_explainer,
)
from .permutation import (
    PERMUTATION_CAPABILITIES,
    PermutationExplainer,
    create_permutation_explainer,
)
from .protocol import (
    FeatureExplainer,
    GradientModelProtocol,
    PredictorProtocol,
    RegressionFeatureExplainer,
    RegressionGradientModelProtocol,
    RegressorPredictorProtocol,
)
from .regression_permutation import (
    REGRESSION_PERMUTATION_CAPABILITIES,
    RegressionPermutationExplainer,
    create_regression_permutation_explainer,
)
from .tree import (
    LocalExplanation,
    ShapTreeWrapper,
    TreeModelProtocol,
)
from .types import (
    ComputationalCost,
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
    GradientConfig,
    IntegratedGradientsConfig,
    PermutationConfig,
)

__all__ = [
    "GRADIENT_CAPABILITIES",
    "INTEGRATED_GRADIENTS_CAPABILITIES",
    "PERMUTATION_CAPABILITIES",
    "REGRESSION_PERMUTATION_CAPABILITIES",
    "ComputationalCost",
    "ExplainerCapabilities",
    "ExplainerName",
    "FeatureExplainer",
    "FeatureImportanceScore",
    "GradientConfig",
    "GradientExplainer",
    "GradientModelProtocol",
    "IntegratedGradientsConfig",
    "IntegratedGradientsExplainer",
    "LocalExplanation",
    "PermutationConfig",
    "PermutationExplainer",
    "PredictorProtocol",
    "RegressionFeatureExplainer",
    "RegressionGradientModelProtocol",
    "RegressionPermutationExplainer",
    "RegressorPredictorProtocol",
    "ShapTreeWrapper",
    "TreeModelProtocol",
    "create_gradient_explainer",
    "create_integrated_gradients_explainer",
    "create_permutation_explainer",
    "create_regression_permutation_explainer",
]
