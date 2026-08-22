"""Regression explainer adapters: gradient, integrated gradients, SHAP tree."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_ml.explainers.protocol import (
    RegressorPredictorProtocol,
)
from platform_ml.explainers.types import (
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
)


class _RegressionGradientModelWrapper:
    """Wrapper to safely access compute_regression_gradients via getattr.

    Used by gradient-based regression explainers to call
    compute_regression_gradients on models that may implement
    RegressionGradientModelProtocol.
    """

    def __init__(self, model: RegressorPredictorProtocol) -> None:
        """Initialize wrapper.

        Args:
            model: Model implementing RegressorPredictorProtocol,
                possibly with compute_regression_gradients method.
        """
        self._model = model

    def compute_regression_gradients(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Delegate to model's compute_regression_gradients method.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Gradients with shape (n_samples, n_features).

        Raises:
            AttributeError: If model doesn't have compute_regression_gradients.
        """
        attr_name = "compute_regression_gradients"
        result: NDArray[np.float64] = getattr(self._model, attr_name)(x)
        return result


def _get_importance_from_pair(pair: tuple[int, float]) -> float:
    """Extract importance value from (index, importance) tuple.

    Args:
        pair: Tuple of (feature_index, importance_score).

    Returns:
        The importance score.
    """
    return pair[1]


def _rank_features(
    feature_names: list[str],
    aggregated: NDArray[np.float64],
) -> list[FeatureImportanceScore]:
    """Rank features by aggregated importance values.

    Args:
        feature_names: List of feature names.
        aggregated: Array of importance values with shape (n_features,).

    Returns:
        Sorted list of FeatureImportanceScore (most important first).
    """
    n_features = len(feature_names)
    index_importance_pairs: list[tuple[int, float]] = []
    for i in range(n_features):
        val: float = float(aggregated.flat[i].item())
        index_importance_pairs.append((i, val))

    sorted_pairs: list[tuple[int, float]] = sorted(
        index_importance_pairs,
        key=_get_importance_from_pair,
        reverse=True,
    )

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


class _RegressionGradientAdapter:
    """Adapter for gradient-based regression feature importance.

    Computes mean absolute gradient of regression output w.r.t. each feature.
    """

    def __init__(self, multiply_by_input: bool, absolute_value: bool) -> None:
        """Initialize adapter.

        Args:
            multiply_by_input: Whether to multiply gradients by input values.
            absolute_value: Whether to take absolute values of attributions.
        """
        self._multiply_by_input = multiply_by_input
        self._absolute_value = absolute_value

    def explainer_name(self) -> ExplainerName:
        """Return explainer name."""
        return "gradient"

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities."""
        return {
            "requires_gradients": True,
            "requires_background_data": False,
            "computational_cost": "low",
        }

    def compute_importance(
        self,
        *,
        model: RegressorPredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
    ) -> list[FeatureImportanceScore]:
        """Compute regression feature importance using gradients.

        Args:
            model: Regression model with compute_regression_gradients method.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.

        Returns:
            List of FeatureImportanceScore sorted by importance.

        Raises:
            AttributeError: If model doesn't have compute_regression_gradients.
        """
        grad_model = _RegressionGradientModelWrapper(model)
        grads: NDArray[np.float64] = grad_model.compute_regression_gradients(x_data)

        if self._multiply_by_input:
            grads = grads * x_data

        n_samples = int(x_data.shape[0])
        summed: NDArray[np.float64] = np.sum(grads, axis=0)
        mean_grads: NDArray[np.float64] = summed / float(n_samples)

        if self._absolute_value:
            mean_grads = np.abs(mean_grads)

        return _rank_features(feature_names, mean_grads)


class _RegressionIntegratedGradientsAdapter:
    """Adapter for integrated gradients regression feature importance."""

    def __init__(self, n_steps: int, baseline_mode: str) -> None:
        """Initialize adapter.

        Args:
            n_steps: Number of interpolation steps.
            baseline_mode: How to compute baseline ("zeros" or "mean").
        """
        self._n_steps = n_steps
        self._baseline_mode = baseline_mode

    def explainer_name(self) -> ExplainerName:
        """Return explainer name."""
        return "integrated_gradients"

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities."""
        return {
            "requires_gradients": True,
            "requires_background_data": False,
            "computational_cost": "high",
        }

    def compute_importance(
        self,
        *,
        model: RegressorPredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
    ) -> list[FeatureImportanceScore]:
        """Compute regression feature importance using integrated gradients.

        Args:
            model: Regression model with compute_regression_gradients method.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.

        Returns:
            List of FeatureImportanceScore sorted by importance.

        Raises:
            AttributeError: If model doesn't have compute_regression_gradients.
        """
        grad_model = _RegressionGradientModelWrapper(model)

        n_samples = int(x_data.shape[0])
        n_features = int(x_data.shape[1])

        baseline: NDArray[np.float64]
        if self._baseline_mode == "zeros":
            baseline = np.zeros((n_samples, n_features), dtype=np.float64)
        else:  # "mean"
            mean_vals: NDArray[np.float64] = np.mean(x_data, axis=0).astype(np.float64)
            baseline = np.tile(mean_vals, (n_samples, 1)).astype(np.float64)

        integrated: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)

        for step in range(self._n_steps):
            alpha = float(step) / float(self._n_steps)
            interpolated: NDArray[np.float64] = (baseline + alpha * (x_data - baseline)).astype(
                np.float64
            )
            grads: NDArray[np.float64] = grad_model.compute_regression_gradients(interpolated)
            integrated = integrated + grads

        diff: NDArray[np.float64] = (x_data - baseline).astype(np.float64)
        integrated = (integrated * diff / float(self._n_steps)).astype(np.float64)

        abs_integrated: NDArray[np.float64] = np.abs(integrated)
        summed: NDArray[np.float64] = np.sum(abs_integrated, axis=0)
        mean_ig: NDArray[np.float64] = (summed / float(n_samples)).astype(np.float64)

        return _rank_features(feature_names, mean_ig)


def _unwrap_for_shap(
    model: RegressorPredictorProtocol,
) -> RegressorPredictorProtocol:
    """Get raw tree model from PreparedRegressor wrapper.

    SHAP TreeExplainer requires raw tree model objects (XGBRegressor,
    LGBMBooster), not PreparedRegressor wrappers. Wrappers expose
    raw_model property containing the underlying tree model.

    Args:
        model: Tree-based regressor (raw or wrapped).

    Returns:
        Raw model for SHAP, or model itself if not wrapped.
    """
    raw_model_attr = "raw_model"
    if not hasattr(model, raw_model_attr):
        return model
    result: RegressorPredictorProtocol = getattr(model, raw_model_attr)
    return result


class _ShapExplainerProtocol(Protocol):
    """Protocol for shap.TreeExplainer for regression models."""

    def shap_values(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute SHAP values for regression model.

        Args:
            x: Input data with shape (n_samples, n_features).

        Returns:
            SHAP values with shape (n_samples, n_features).
        """
        ...


class _ShapTreeExplainerCtor(Protocol):
    """Protocol for shap.TreeExplainer constructor."""

    def __call__(
        self,
        model: RegressorPredictorProtocol,
    ) -> _ShapExplainerProtocol:
        """Create a TreeExplainer for a tree-based regressor.

        Args:
            model: Tree-based regressor model.

        Returns:
            SHAP TreeExplainer instance.
        """
        ...


class _RegressionShapTreeAdapter:
    """Adapter for SHAP tree-based regression feature importance.

    Works with XGBoost and LightGBM regressors. SHAP TreeExplainer
    handles regressors natively (single output, no class selection).

    Automatically unwraps PreparedRegressor wrappers that expose
    raw_model, since SHAP requires the raw tree model object.
    """

    def explainer_name(self) -> ExplainerName:
        """Return explainer name."""
        return "shap_tree"

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities."""
        return {
            "requires_gradients": False,
            "requires_background_data": False,
            "computational_cost": "medium",
        }

    def compute_importance(
        self,
        *,
        model: RegressorPredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
    ) -> list[FeatureImportanceScore]:
        """Compute regression feature importance using SHAP TreeExplainer.

        Unwraps PreparedRegressor wrappers that expose raw_model,
        since SHAP TreeExplainer requires raw tree model objects
        (XGBRegressor, LGBMBooster), not wrapper classes.

        Args:
            model: Tree-based regressor model (raw or wrapped).
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.

        Returns:
            List of FeatureImportanceScore sorted by importance.
        """
        shap_model = _unwrap_for_shap(model)

        shap_mod = __import__("shap")
        tree_explainer_ctor: _ShapTreeExplainerCtor = shap_mod.TreeExplainer
        explainer: _ShapExplainerProtocol = tree_explainer_ctor(shap_model)

        shap_values: NDArray[np.float64] = explainer.shap_values(x_data)

        abs_shap: NDArray[np.float64] = np.abs(shap_values)
        n_samples = int(abs_shap.shape[0])
        summed: NDArray[np.float64] = np.sum(abs_shap, axis=0)
        aggregated: NDArray[np.float64] = (summed / float(n_samples)).astype(np.float64)

        return _rank_features(feature_names, aggregated)


__all__ = [
    "_RegressionGradientAdapter",
    "_RegressionGradientModelWrapper",
    "_RegressionIntegratedGradientsAdapter",
    "_RegressionShapTreeAdapter",
    "_rank_features",
    "_unwrap_for_shap",
]
