"""Regression permutation importance explainer.

Computes feature importance by measuring prediction change when shuffling features.
Uses MSE change instead of probability change (classifier version).
Model-agnostic: works with any model implementing RegressorPredictorProtocol.

Algorithm:
1. Compute baseline predictions on original data
2. For each feature:
   a. Shuffle the feature column n_repeats times
   b. Compute predictions on shuffled data
   c. Measure mean squared error change
3. Rank features by mean importance across repeats

Higher importance means the model relies more heavily on that feature.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .protocol import RegressorPredictorProtocol
from .types import (
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
    PermutationConfig,
)

REGRESSION_PERMUTATION_CAPABILITIES: ExplainerCapabilities = {
    "requires_gradients": False,
    "requires_background_data": False,
    "computational_cost": "medium",
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


def _compute_baseline_mse(
    model: RegressorPredictorProtocol,
    x_data: NDArray[np.float64],
    y_pred_baseline: NDArray[np.float64],
) -> float:
    """Compute baseline MSE (always 0 since comparing to self).

    Kept for symmetry with classifier version — the actual importance
    is measured as delta MSE when a feature is permuted.

    Args:
        model: Model with predict method.
        x_data: Input features.
        y_pred_baseline: Baseline predictions.

    Returns:
        Always 0.0 (predictions match themselves).
    """
    _ = model
    _ = x_data
    _ = y_pred_baseline
    return 0.0


def _compute_single_feature_importance(
    model: RegressorPredictorProtocol,
    x_data: NDArray[np.float64],
    baseline_preds: NDArray[np.float64],
    feature_idx: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> float:
    """Compute importance for a single feature via MSE change.

    Args:
        model: Model with predict method.
        x_data: Input features.
        baseline_preds: Baseline predictions with shape (n_samples,).
        feature_idx: Index of feature to permute.
        n_repeats: Number of permutation repeats.
        rng: Random number generator.

    Returns:
        Mean squared error increase when feature is shuffled.
    """
    n_samples = int(x_data.shape[0])
    importance_scores: list[float] = []

    for _ in range(n_repeats):
        x_permuted: NDArray[np.float64] = x_data.copy()
        shuffled_indices: NDArray[np.intp] = rng.permutation(n_samples)
        shuffled_column: NDArray[np.float64] = x_data[shuffled_indices, feature_idx]
        x_permuted[:, feature_idx] = shuffled_column

        permuted_preds: NDArray[np.float64] = model.predict(x_permuted)

        diff: NDArray[np.float64] = baseline_preds - permuted_preds
        squared_diff: NDArray[np.float64] = diff * diff
        mse_change: float = float(np.sum(squared_diff)) / n_samples
        importance_scores.append(mse_change)

    return sum(importance_scores) / len(importance_scores)


def _get_importance_from_pair(pair: tuple[str, float]) -> float:
    """Extract importance value from (name, importance) tuple.

    Args:
        pair: Tuple of (feature_name, importance_score).

    Returns:
        The importance score.
    """
    return pair[1]


def _rank_importances(
    feature_names: list[str],
    importances: list[float],
) -> list[FeatureImportanceScore]:
    """Rank features by importance and return sorted list.

    Args:
        feature_names: List of feature names.
        importances: List of importance scores.

    Returns:
        List of FeatureImportanceScore sorted by rank (most important first).
    """
    name_importance_pairs: list[tuple[str, float]] = list(
        zip(feature_names, importances, strict=True)
    )

    sorted_pairs: list[tuple[str, float]] = sorted(
        name_importance_pairs,
        key=_get_importance_from_pair,
        reverse=True,
    )

    results: list[FeatureImportanceScore] = []
    for rank_idx, pair in enumerate(sorted_pairs):
        name: str = pair[0]
        importance: float = pair[1]
        score: FeatureImportanceScore = {
            "name": name,
            "importance": importance,
            "rank": rank_idx + 1,
        }
        results.append(score)

    return results


class RegressionPermutationExplainer:
    """Regression permutation importance explainer.

    Computes feature importance by measuring how much regression predictions
    change (MSE) when each feature is randomly shuffled.

    Args:
        config: Configuration with n_repeats and random_state.
    """

    def __init__(self, config: PermutationConfig) -> None:
        """Initialize regression permutation explainer.

        Args:
            config: Configuration with n_repeats and random_state.
        """
        self._n_repeats = int(config["n_repeats"])
        self._random_state = int(config["random_state"])

    def explainer_name(self) -> ExplainerName:
        """Return explainer name.

        Returns:
            Literal "permutation".
        """
        return "permutation"

    def capabilities(self) -> ExplainerCapabilities:
        """Return explainer capabilities.

        Returns:
            Capabilities indicating no gradients required, medium cost.
        """
        return REGRESSION_PERMUTATION_CAPABILITIES

    def compute_importance(
        self,
        *,
        model: RegressorPredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
    ) -> list[FeatureImportanceScore]:
        """Compute regression permutation feature importance.

        Args:
            model: Model implementing RegressorPredictorProtocol.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names matching x_data columns.

        Returns:
            List of FeatureImportanceScore sorted by rank.

        Raises:
            ValueError: If feature_names length doesn't match x_data columns.
        """
        _validate_inputs(x_data, feature_names)

        rng = np.random.default_rng(self._random_state)
        n_features = int(x_data.shape[1])

        baseline_preds: NDArray[np.float64] = model.predict(x_data)

        importances: list[float] = []
        for feature_idx in range(n_features):
            importance = _compute_single_feature_importance(
                model=model,
                x_data=x_data,
                baseline_preds=baseline_preds,
                feature_idx=feature_idx,
                n_repeats=self._n_repeats,
                rng=rng,
            )
            importances.append(importance)

        return _rank_importances(feature_names, importances)


def create_regression_permutation_explainer(
    config: PermutationConfig,
) -> RegressionPermutationExplainer:
    """Factory function to create a RegressionPermutationExplainer.

    Args:
        config: Configuration with n_repeats and random_state.

    Returns:
        Configured RegressionPermutationExplainer instance.
    """
    return RegressionPermutationExplainer(config)


__all__ = [
    "REGRESSION_PERMUTATION_CAPABILITIES",
    "RegressionPermutationExplainer",
    "create_regression_permutation_explainer",
]
