"""Gradient-based feature importance explainer.

Computes feature importance using input gradients from differentiable models.
Requires models implementing GradientModelProtocol.

Algorithm:
1. Compute gradients of target class output w.r.t. input features
2. Optionally multiply gradients by input values (gradient * input)
3. Aggregate across samples using mean absolute value
4. Rank features by aggregated importance

This is a simple, fast method suitable for neural networks.
For more accurate attributions, use IntegratedGradientsExplainer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .protocol import GradientModelProtocol
from .types import (
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
    GradientConfig,
)

GRADIENT_CAPABILITIES: ExplainerCapabilities = {
    "requires_gradients": True,
    "requires_background_data": False,
    "computational_cost": "low",
}


def _validate_inputs(
    x_data: NDArray[np.float64],
    feature_names: list[str],
) -> None:
    """Validate input data and feature names match.

    Args:
        x_data: Input features array.
        feature_names: List of feature names.

    Raises:
        ValueError: If dimensions don't match.
    """
    n_features = int(x_data.shape[1])
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match x_data columns ({n_features})"
        )


def _compute_attributions(
    model: GradientModelProtocol,
    x_data: NDArray[np.float64],
    target_class: int,
    multiply_by_input: bool,
    absolute_value: bool,
) -> NDArray[np.float64]:
    """Compute gradient-based attributions.

    Args:
        model: Model with compute_gradients method.
        x_data: Input features.
        target_class: Class index for gradient computation.
        multiply_by_input: Whether to multiply gradients by inputs.
        absolute_value: Whether to take absolute values.

    Returns:
        Attributions with shape (n_samples, n_features).
    """
    gradients: NDArray[np.float64] = model.compute_gradients(x_data, target_class)

    attributions: NDArray[np.float64] = gradients * x_data if multiply_by_input else gradients

    if absolute_value:
        attributions = np.abs(attributions)

    return attributions.astype(np.float64)


def _aggregate_attributions(
    attributions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Aggregate attributions across samples.

    Args:
        attributions: Attributions with shape (n_samples, n_features).

    Returns:
        Mean absolute attribution per feature with shape (n_features,).
    """
    n_samples = int(attributions.shape[0])
    # Use np.abs with type annotation, then sum/len for mean
    abs_attr: NDArray[np.float64] = np.abs(attributions)
    summed: NDArray[np.float64] = np.sum(abs_attr, axis=0)
    mean_attr: NDArray[np.float64] = summed / float(n_samples)
    return mean_attr


def _get_importance_from_pair(pair: tuple[int, float]) -> float:
    """Extract importance value from (index, importance) tuple.

    Args:
        pair: Tuple of (feature_index, importance_score).

    Returns:
        The importance score.
    """
    return pair[1]


def _rank_importances(
    feature_names: list[str],
    importances: NDArray[np.float64],
) -> list[FeatureImportanceScore]:
    """Rank features by importance and return sorted list.

    Args:
        feature_names: List of feature names.
        importances: Array of importance scores.

    Returns:
        List of FeatureImportanceScore sorted by rank (most important first).
    """
    # Build list of (index, importance) using flat iterator with item()
    index_importance_pairs: list[tuple[int, float]] = []
    for i, imp in enumerate(importances.flat):
        imp_float: float = float(imp.item())
        index_importance_pairs.append((i, imp_float))

    # Sort by importance descending using typed helper
    sorted_pairs: list[tuple[int, float]] = sorted(
        index_importance_pairs,
        key=_get_importance_from_pair,
        reverse=True,
    )

    # Build ranked results
    results: list[FeatureImportanceScore] = []
    for rank_idx, pair in enumerate(sorted_pairs):
        feature_idx: int = pair[0]
        imp_value: float = pair[1]
        score: FeatureImportanceScore = {
            "name": feature_names[feature_idx],
            "importance": imp_value,
            "rank": rank_idx + 1,
        }
        results.append(score)

    return results


class GradientExplainer:
    """Gradient-based feature importance explainer.

    Computes feature importance using gradients of model output w.r.t. inputs.
    Fast but less accurate than integrated gradients.

    Args:
        config: Configuration with multiply_by_input and absolute_value flags.

    Example:
        >>> config: GradientConfig = {"multiply_by_input": True, "absolute_value": True}
        >>> explainer = GradientExplainer(config)
        >>> importance = explainer.compute_importance(
        ...     model=gradient_model,
        ...     x_data=x_test,
        ...     feature_names=["feat1", "feat2"],
        ...     target_class=1,
        ... )
    """

    def __init__(self, config: GradientConfig) -> None:
        """Initialize gradient explainer.

        Args:
            config: Configuration with multiply_by_input and absolute_value flags.
        """
        self._multiply_by_input = bool(config["multiply_by_input"])
        self._absolute_value = bool(config["absolute_value"])

    def explainer_name(self) -> ExplainerName:
        """Return explainer name.

        Returns:
            Literal "gradient".
        """
        return "gradient"

    def capabilities(self) -> ExplainerCapabilities:
        """Return explainer capabilities.

        Returns:
            Capabilities indicating gradients required, low cost.
        """
        return GRADIENT_CAPABILITIES

    def compute_importance(
        self,
        *,
        model: GradientModelProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute gradient-based feature importance.

        Args:
            model: Model implementing GradientModelProtocol.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names matching x_data columns.
            target_class: Class index for which to compute importance.

        Returns:
            List of FeatureImportanceScore sorted by rank.

        Raises:
            ValueError: If feature_names length doesn't match x_data columns.
        """
        _validate_inputs(x_data, feature_names)

        attributions = _compute_attributions(
            model=model,
            x_data=x_data,
            target_class=target_class,
            multiply_by_input=self._multiply_by_input,
            absolute_value=self._absolute_value,
        )

        importances = _aggregate_attributions(attributions)

        return _rank_importances(feature_names, importances)


def create_gradient_explainer(config: GradientConfig) -> GradientExplainer:
    """Factory function to create a GradientExplainer.

    Args:
        config: Configuration with multiply_by_input and absolute_value flags.

    Returns:
        Configured GradientExplainer instance.
    """
    return GradientExplainer(config)


__all__ = [
    "GRADIENT_CAPABILITIES",
    "GradientExplainer",
    "create_gradient_explainer",
]
