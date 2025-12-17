"""Protocol definitions for feature importance explainers.

Defines the FeatureExplainer protocol that all concrete explainers implement.
Also defines supporting protocols for model interfaces.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .types import ExplainerCapabilities, ExplainerName, FeatureImportanceScore


class PredictorProtocol(Protocol):
    """Protocol for models that can make predictions.

    This is the minimal interface required by permutation importance.
    Any model with a predict_proba method satisfies this protocol.
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        ...


class GradientModelProtocol(Protocol):
    """Protocol for differentiable models that support gradient computation.

    Extends PredictorProtocol with gradient computation capabilities.
    Required by gradient-based explainers (GradientExplainer, IntegratedGradientsExplainer).
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class probabilities for input samples.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Class probabilities with shape (n_samples, n_classes).
        """
        ...

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input features.

        Args:
            x: Input features with shape (n_samples, n_features).
            target_class: Class index for which to compute gradients.

        Returns:
            Gradients with shape (n_samples, n_features).
        """
        ...


class FeatureExplainer(Protocol):
    """Protocol for feature importance explainers.

    All concrete explainers (PermutationExplainer, GradientExplainer, etc.)
    implement this protocol.
    """

    def explainer_name(self) -> ExplainerName:
        """Return the name of this explainer.

        Returns:
            One of: "permutation", "gradient", "integrated_gradients".
        """
        ...

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities of this explainer.

        Returns:
            TypedDict describing what the explainer requires and its cost.
        """
        ...

    def compute_importance(
        self,
        *,
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute feature importance scores.

        Args:
            model: Model implementing PredictorProtocol (or GradientModelProtocol
                for gradient-based explainers).
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names matching x_data columns.
            target_class: Class index for which to compute importance.

        Returns:
            List of FeatureImportanceScore, sorted by rank (most important first).

        Raises:
            ValueError: If feature_names length doesn't match x_data columns.
        """
        ...


__all__ = [
    "FeatureExplainer",
    "GradientModelProtocol",
    "PredictorProtocol",
]
