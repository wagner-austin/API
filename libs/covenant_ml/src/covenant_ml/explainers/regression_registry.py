"""Regression explainer registry with regressor backend compatibility.

Provides a registry of regression feature importance explainers with explicit
backend compatibility. Separate from classifier registry because regression
uses predict() instead of predict_proba() and has no target_class.

Supported explainers:
- permutation: All regressor backends (model-agnostic, MSE-based)
- shap_tree: Tree regressors only (xgboost_reg, lightgbm_reg)
- gradient: Neural regressors only (mlp_reg, lstm_reg)
- integrated_gradients: Neural regressors only (mlp_reg, lstm_reg)

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_ml.explainers import (
    PermutationConfig,
    create_regression_permutation_explainer,
)
from platform_ml.explainers.protocol import (
    RegressionFeatureExplainer,
    RegressorPredictorProtocol,
)
from platform_ml.explainers.types import (
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
)

from ..types import RegressorBackendName
from .types import SupportedExplainer

# ---------------------------------------------------------------------------
# Gradient model wrapper for regression
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Ranking helper (shared by gradient/IG/shap adapters)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Regression Gradient Adapter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Regression Integrated Gradients Adapter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Regression SHAP Tree Adapter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RegressionExplainerFactory(Protocol):
    """Factory protocol to construct a regression explainer."""

    def __call__(self) -> RegressionFeatureExplainer:
        """Create and return a regression explainer instance."""
        ...


class RegressionExplainerRegistration:
    """Registration record for a regression explainer.

    Args:
        factory: Callable that creates a regression explainer instance.
        compatible_backends: Set of regressor backends this explainer works with.
        requires_gradients: True if explainer needs compute_regression_gradients.
    """

    def __init__(
        self,
        factory: RegressionExplainerFactory,
        compatible_backends: frozenset[RegressorBackendName],
        requires_gradients: bool,
    ) -> None:
        """Initialize registration.

        Args:
            factory: Callable that creates a regression explainer instance.
            compatible_backends: Set of regressor backends this works with.
            requires_gradients: True if explainer needs gradient computation.
        """
        self._factory = factory
        self._compatible_backends = compatible_backends
        self._requires_gradients = requires_gradients

    def factory(self) -> RegressionExplainerFactory:
        """Return the factory function."""
        return self._factory

    def compatible_backends(self) -> frozenset[RegressorBackendName]:
        """Return set of compatible regressor backend names."""
        return self._compatible_backends

    def requires_gradients(self) -> bool:
        """Return True if explainer requires gradient computation."""
        return self._requires_gradients


class RegressionExplainerRegistry:
    """Registry of regression explainers keyed by name with backend compatibility.

    Provides methods to register explainers, query compatibility,
    and instantiate regression explainers by name.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._map: dict[SupportedExplainer, RegressionExplainerRegistration] = {}

    def register(
        self,
        name: SupportedExplainer,
        registration: RegressionExplainerRegistration,
    ) -> None:
        """Register a regression explainer.

        Args:
            name: Unique explainer name.
            registration: Registration with factory and compatibility info.
        """
        self._map[name] = registration

    def list_explainers(self) -> list[SupportedExplainer]:
        """Return sorted list of registered explainer names.

        Returns:
            List of explainer names in alphabetical order.
        """
        names: list[SupportedExplainer] = list(self._map.keys())
        return sorted(names)

    def list_compatible_explainers(
        self,
        backend: RegressorBackendName,
    ) -> list[SupportedExplainer]:
        """List explainers compatible with a given regressor backend.

        Args:
            backend: Regressor backend name to check compatibility for.

        Returns:
            Sorted list of compatible explainer names.
        """
        compatible: list[SupportedExplainer] = []
        for name, reg in self._map.items():
            if backend in reg.compatible_backends():
                compatible.append(name)
        return sorted(compatible)

    def get(self, name: SupportedExplainer) -> RegressionFeatureExplainer:
        """Create and return a regression explainer instance.

        Args:
            name: Name of explainer to instantiate.

        Returns:
            New regression explainer instance.

        Raises:
            KeyError: If explainer name is not registered.
        """
        reg = self._map[name]
        return reg.factory()()

    def is_compatible(
        self,
        explainer: SupportedExplainer,
        backend: RegressorBackendName,
    ) -> bool:
        """Check if explainer is compatible with a regressor backend.

        Args:
            explainer: Explainer name to check.
            backend: Regressor backend name to check compatibility for.

        Returns:
            True if explainer can be used with backend, False otherwise.
        """
        reg = self._map.get(explainer)
        if reg is None:
            return False
        return backend in reg.compatible_backends()


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _create_regression_permutation_factory() -> RegressionExplainerFactory:
    """Create factory for regression permutation explainer."""

    def factory() -> RegressionFeatureExplainer:
        config: PermutationConfig = {"n_repeats": 10, "random_state": 42}
        return create_regression_permutation_explainer(config)

    return factory


def _create_regression_gradient_factory() -> RegressionExplainerFactory:
    """Create factory for regression gradient explainer."""

    def factory() -> RegressionFeatureExplainer:
        return _RegressionGradientAdapter(multiply_by_input=True, absolute_value=True)

    return factory


def _create_regression_integrated_gradients_factory() -> RegressionExplainerFactory:
    """Create factory for regression integrated gradients explainer."""

    def factory() -> RegressionFeatureExplainer:
        return _RegressionIntegratedGradientsAdapter(n_steps=50, baseline_mode="zeros")

    return factory


def _create_regression_shap_tree_factory() -> RegressionExplainerFactory:
    """Create factory for regression SHAP tree explainer."""

    def factory() -> RegressionFeatureExplainer:
        return _RegressionShapTreeAdapter()

    return factory


def default_regression_explainer_registry() -> RegressionExplainerRegistry:
    """Build the default registry with all supported regression explainers.

    Returns:
        RegressionExplainerRegistry with permutation, gradient,
        integrated_gradients, and shap_tree explainers registered.
    """
    reg = RegressionExplainerRegistry()

    # Permutation: works with all regressor backends (model-agnostic)
    reg.register(
        "permutation",
        RegressionExplainerRegistration(
            factory=_create_regression_permutation_factory(),
            compatible_backends=frozenset(
                [
                    "xgboost_reg",
                    "lightgbm_reg",
                    "mlp_reg",
                    "lstm_reg",
                ]
            ),
            requires_gradients=False,
        ),
    )

    # Gradient: only neural regressor backends
    reg.register(
        "gradient",
        RegressionExplainerRegistration(
            factory=_create_regression_gradient_factory(),
            compatible_backends=frozenset(["mlp_reg", "lstm_reg"]),
            requires_gradients=True,
        ),
    )

    # Integrated Gradients: only neural regressor backends
    reg.register(
        "integrated_gradients",
        RegressionExplainerRegistration(
            factory=_create_regression_integrated_gradients_factory(),
            compatible_backends=frozenset(["mlp_reg", "lstm_reg"]),
            requires_gradients=True,
        ),
    )

    # SHAP Tree: tree-based regressor backends
    reg.register(
        "shap_tree",
        RegressionExplainerRegistration(
            factory=_create_regression_shap_tree_factory(),
            compatible_backends=frozenset(["xgboost_reg", "lightgbm_reg"]),
            requires_gradients=False,
        ),
    )

    return reg


__all__ = [
    "RegressionExplainerFactory",
    "RegressionExplainerRegistration",
    "RegressionExplainerRegistry",
    "_RegressionGradientAdapter",
    "_RegressionGradientModelWrapper",
    "_RegressionIntegratedGradientsAdapter",
    "_RegressionShapTreeAdapter",
    "_rank_features",
    "_unwrap_for_shap",
    "default_regression_explainer_registry",
]
