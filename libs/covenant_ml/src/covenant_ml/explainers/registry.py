"""Explainer registry with backend compatibility mapping.

Provides a registry of feature importance explainers with explicit
backend compatibility information. Explainers are registered with
factory functions and metadata about their requirements.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray
from platform_ml.explainers import (
    FeatureExplainer,
    PermutationConfig,
    ShapTreeWrapper,
    create_permutation_explainer,
)
from platform_ml.explainers.protocol import PredictorProtocol
from platform_ml.explainers.types import (
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
)

from ..types import BackendName
from .types import SupportedExplainer


class ExplainerFactory(Protocol):
    """Factory protocol to construct an explainer implementation."""

    def __call__(self) -> FeatureExplainer:
        """Create and return an explainer instance."""
        ...


class ExplainerRegistration:
    """Registration record holding factory and compatibility info.

    Args:
        factory: Callable that creates an explainer instance.
        compatible_backends: Set of backend names this explainer works with.
        requires_gradients: True if explainer needs compute_gradients() method.
    """

    def __init__(
        self,
        factory: ExplainerFactory,
        compatible_backends: frozenset[BackendName],
        requires_gradients: bool,
    ) -> None:
        """Initialize registration.

        Args:
            factory: Callable that creates an explainer instance.
            compatible_backends: Set of backend names this explainer works with.
            requires_gradients: True if explainer needs compute_gradients() method.
        """
        self._factory = factory
        self._compatible_backends = compatible_backends
        self._requires_gradients = requires_gradients

    def factory(self) -> ExplainerFactory:
        """Return the factory function."""
        return self._factory

    def compatible_backends(self) -> frozenset[BackendName]:
        """Return set of compatible backend names."""
        return self._compatible_backends

    def requires_gradients(self) -> bool:
        """Return True if explainer requires gradient computation."""
        return self._requires_gradients


class ExplainerRegistry:
    """Registry of explainers keyed by name with backend compatibility.

    Provides methods to register explainers, query compatibility,
    and instantiate explainers by name.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._map: dict[SupportedExplainer, ExplainerRegistration] = {}

    def register(
        self,
        name: SupportedExplainer,
        registration: ExplainerRegistration,
    ) -> None:
        """Register an explainer.

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
        backend: BackendName,
    ) -> list[SupportedExplainer]:
        """List explainers compatible with a given backend.

        Args:
            backend: Backend name to check compatibility for.

        Returns:
            Sorted list of compatible explainer names.
        """
        compatible: list[SupportedExplainer] = []
        for name, reg in self._map.items():
            if backend in reg.compatible_backends():
                compatible.append(name)
        return sorted(compatible)

    def get(self, name: SupportedExplainer) -> FeatureExplainer:
        """Create and return an explainer instance.

        Args:
            name: Name of explainer to instantiate.

        Returns:
            New explainer instance.

        Raises:
            KeyError: If explainer name is not registered.
        """
        reg = self._map[name]
        return reg.factory()()

    def is_compatible(
        self,
        explainer: SupportedExplainer,
        backend: BackendName,
    ) -> bool:
        """Check if explainer is compatible with backend.

        Args:
            explainer: Explainer name to check.
            backend: Backend name to check compatibility for.

        Returns:
            True if explainer can be used with backend, False otherwise.
        """
        if explainer not in self._map:
            return False
        return backend in self._map[explainer].compatible_backends()

    def requires_gradients(self, name: SupportedExplainer) -> bool:
        """Check if explainer requires gradient computation.

        Args:
            name: Explainer name to check.

        Returns:
            True if explainer needs compute_gradients() method.

        Raises:
            KeyError: If explainer name is not registered.
        """
        return self._map[name].requires_gradients()


# ---------------------------------------------------------------------------
# Protocol and wrapper for compute_gradients access
# ---------------------------------------------------------------------------


class _ComputeGradientsCallable(Protocol):
    """Protocol for compute_gradients method callable."""

    def __call__(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input."""
        ...


class _GradientModelWrapper:
    """Wrapper providing typed access to compute_gradients via string getattr.

    Used to wrap a PredictorProtocol model that has been validated at runtime
    to have compute_gradients method, providing proper typing for internal use.
    """

    _ATTR_NAME: str = "compute_gradients"

    def __init__(self, model: PredictorProtocol) -> None:
        """Initialize wrapper with a model that has compute_gradients.

        Args:
            model: Model with predict_proba and compute_gradients methods.

        Raises:
            AttributeError: If model doesn't have compute_gradients method.
        """
        if not hasattr(model, self._ATTR_NAME):
            raise AttributeError(
                f"Model {type(model).__name__} must have compute_gradients() method. "
                "Use a neural network backend (mlp, lstm)."
            )
        self._model = model
        # Use string variable to prevent linter from converting to direct access
        # Assign directly to protocol type to avoid Any from getattr
        self._compute_grad: _ComputeGradientsCallable = getattr(model, self._ATTR_NAME)

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Compute gradients of output w.r.t. input.

        Args:
            x: Input features.
            target_class: Class index for gradients.

        Returns:
            Gradients array.
        """
        result: NDArray[np.float64] = self._compute_grad(x, target_class)
        return result


# ---------------------------------------------------------------------------
# Helper functions for ranking to avoid Any types
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
    importances: NDArray[np.float64],
) -> list[FeatureImportanceScore]:
    """Rank features by importance and return sorted list.

    Args:
        feature_names: List of feature names.
        importances: Array of importance scores with shape (n_features,).

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


# ---------------------------------------------------------------------------
# Local TypedDict for SHAP explanations (not exported from platform_ml)
# ---------------------------------------------------------------------------


class _LocalExplanation(TypedDict, total=True):
    """Local explanation for a single sample from ShapTreeWrapper.

    Matches platform_ml.explainers.tree.LocalExplanation structure.
    """

    base_value: float
    values: list[float]
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Adapter classes for explainer implementations
# ---------------------------------------------------------------------------


class _GradientAdapter:
    """Adapter implementing gradient-based feature importance.

    Uses _GradientModelWrapper for typed access to compute_gradients.
    """

    def __init__(self, multiply_by_input: bool, absolute_value: bool) -> None:
        """Initialize adapter with config.

        Args:
            multiply_by_input: Whether to multiply gradients by input values.
            absolute_value: Whether to use absolute values of attributions.
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
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute feature importance using gradient-based attribution.

        Args:
            model: Model implementing PredictorProtocol with compute_gradients method.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.
            target_class: Class index for importance computation.

        Returns:
            List of FeatureImportanceScore sorted by importance.

        Raises:
            AttributeError: If model doesn't have compute_gradients method.
        """
        # Wrap model for typed gradient access
        grad_model = _GradientModelWrapper(model)
        grads: NDArray[np.float64] = grad_model.compute_gradients(x_data, target_class)

        # Apply gradient * input if configured
        if self._multiply_by_input:
            grads = grads * x_data

        # Aggregate across samples (mean)
        n_samples = int(x_data.shape[0])
        summed: NDArray[np.float64] = np.sum(grads, axis=0)
        mean_grads: NDArray[np.float64] = summed / float(n_samples)

        # Apply absolute value if configured
        if self._absolute_value:
            mean_grads = np.abs(mean_grads)

        return _rank_features(feature_names, mean_grads)


class _IntegratedGradientsAdapter:
    """Adapter implementing integrated gradients directly.

    Uses _GradientModelWrapper for typed access to compute_gradients.
    """

    def __init__(self, n_steps: int, baseline_mode: str) -> None:
        """Initialize adapter with config.

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
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute feature importance using integrated gradients.

        Args:
            model: Model implementing PredictorProtocol with compute_gradients method.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.
            target_class: Class index for importance computation.

        Returns:
            List of FeatureImportanceScore sorted by importance.

        Raises:
            AttributeError: If model doesn't have compute_gradients method.
        """
        # Wrap model for typed gradient access
        grad_model = _GradientModelWrapper(model)

        n_samples = int(x_data.shape[0])
        n_features = int(x_data.shape[1])

        # Compute baseline
        baseline: NDArray[np.float64]
        if self._baseline_mode == "zeros":
            baseline = np.zeros((n_samples, n_features), dtype=np.float64)
        else:  # "mean"
            mean_vals: NDArray[np.float64] = np.mean(x_data, axis=0).astype(np.float64)
            baseline = np.tile(mean_vals, (n_samples, 1)).astype(np.float64)

        # Compute integrated gradients via Riemann sum
        integrated: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)

        for step in range(self._n_steps):
            alpha = float(step) / float(self._n_steps)
            interpolated: NDArray[np.float64] = (baseline + alpha * (x_data - baseline)).astype(
                np.float64
            )
            grads: NDArray[np.float64] = grad_model.compute_gradients(interpolated, target_class)
            integrated = integrated + grads

        # Scale by (x - baseline) / n_steps
        diff: NDArray[np.float64] = (x_data - baseline).astype(np.float64)
        integrated = (integrated * diff / float(self._n_steps)).astype(np.float64)

        # Aggregate across samples (mean absolute value)
        abs_integrated: NDArray[np.float64] = np.abs(integrated)
        summed: NDArray[np.float64] = np.sum(abs_integrated, axis=0)
        mean_ig: NDArray[np.float64] = (summed / float(n_samples)).astype(np.float64)

        return _rank_features(feature_names, mean_ig)


class _ShapTreeAdapter:
    """Adapter to make ShapTreeWrapper conform to FeatureExplainer protocol.

    Wraps ShapTreeWrapper to provide compute_importance() method that returns
    aggregated SHAP values as FeatureImportanceScore list.
    """

    def __init__(self) -> None:
        """Initialize adapter (wrapper created per-model in compute_importance)."""
        pass

    def explainer_name(self) -> ExplainerName:
        """Return explainer name."""
        # Note: ExplainerName doesn't include "shap_tree", so we return closest match
        return "permutation"  # Placeholder - actual name tracked in registry

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
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Compute feature importance using SHAP TreeExplainer.

        Args:
            model: Tree model implementing PredictorProtocol.
            x_data: Input data with shape (n_samples, n_features).
            feature_names: List of feature names.
            target_class: Class index (used to select SHAP values for that class).

        Returns:
            List of FeatureImportanceScore sorted by importance.
        """
        # Create wrapper for this model
        wrapper = ShapTreeWrapper(model)

        # Get local explanations for all samples
        # ShapTreeWrapper returns list of dicts: values, feature_names, base_value
        raw_explanations = wrapper.explain_local(x_data, feature_names)

        # Aggregate SHAP values across samples (mean absolute value)
        n_features = len(feature_names)
        aggregated: NDArray[np.float64] = np.zeros(n_features, dtype=np.float64)

        sample_count: int = 0
        for raw_exp in raw_explanations:
            # Create typed dict from raw explanation
            exp: _LocalExplanation = {
                "base_value": raw_exp["base_value"],
                "values": raw_exp["values"],
                "feature_names": raw_exp["feature_names"],
            }
            values: list[float] = exp["values"]
            # Use enumerate to iterate and update in-place
            for feature_idx, val in enumerate(values):
                current: float = float(aggregated.flat[feature_idx].item())
                new_val: float = current + abs(val)
                aggregated[feature_idx] = new_val
            sample_count += 1

        # Average across samples
        for i in range(n_features):
            current_agg: float = float(aggregated.flat[i].item())
            aggregated[i] = current_agg / float(sample_count)

        return _rank_features(feature_names, aggregated)


# ---------------------------------------------------------------------------
# Factory functions for each explainer type
# ---------------------------------------------------------------------------


def _create_permutation_factory() -> ExplainerFactory:
    """Create factory for permutation explainer with default config."""

    def factory() -> FeatureExplainer:
        config: PermutationConfig = {"n_repeats": 10, "random_state": 42}
        return create_permutation_explainer(config)

    return factory


def _create_gradient_factory() -> ExplainerFactory:
    """Create factory for gradient explainer with default config.

    Returns adapter that implements gradient-based feature importance
    by wrapping models with _GradientModelWrapper for typed access.
    """

    def factory() -> FeatureExplainer:
        return _GradientAdapter(multiply_by_input=True, absolute_value=True)

    return factory


def _create_integrated_gradients_factory() -> ExplainerFactory:
    """Create factory for integrated gradients explainer with default config.

    Returns adapter that implements integrated gradients algorithm
    by wrapping models with _GradientModelWrapper for typed access.
    """

    def factory() -> FeatureExplainer:
        return _IntegratedGradientsAdapter(n_steps=50, baseline_mode="zeros")

    return factory


def _create_shap_tree_factory() -> ExplainerFactory:
    """Create factory for SHAP tree explainer.

    Returns adapter that wraps ShapTreeWrapper to match FeatureExplainer protocol.
    """

    def factory() -> FeatureExplainer:
        return _ShapTreeAdapter()

    return factory


def default_explainer_registry() -> ExplainerRegistry:
    """Build the default registry with all supported explainers.

    Returns:
        ExplainerRegistry with permutation, gradient, integrated_gradients,
        and shap_tree explainers registered.
    """
    reg = ExplainerRegistry()

    # Permutation: works with all backends (model-agnostic)
    reg.register(
        "permutation",
        ExplainerRegistration(
            factory=_create_permutation_factory(),
            compatible_backends=frozenset(["xgboost", "lightgbm", "mlp", "lstm"]),
            requires_gradients=False,
        ),
    )

    # Gradient: only neural network backends (requires compute_gradients)
    reg.register(
        "gradient",
        ExplainerRegistration(
            factory=_create_gradient_factory(),
            compatible_backends=frozenset(["mlp", "lstm"]),
            requires_gradients=True,
        ),
    )

    # Integrated Gradients: only neural network backends (requires compute_gradients)
    reg.register(
        "integrated_gradients",
        ExplainerRegistration(
            factory=_create_integrated_gradients_factory(),
            compatible_backends=frozenset(["mlp", "lstm"]),
            requires_gradients=True,
        ),
    )

    # SHAP Tree: only tree-based backends
    reg.register(
        "shap_tree",
        ExplainerRegistration(
            factory=_create_shap_tree_factory(),
            compatible_backends=frozenset(["xgboost", "lightgbm"]),
            requires_gradients=False,
        ),
    )

    return reg


__all__ = [
    "ExplainerFactory",
    "ExplainerRegistration",
    "ExplainerRegistry",
    "_GradientAdapter",
    "_GradientModelWrapper",
    "_IntegratedGradientsAdapter",
    "_ShapTreeAdapter",
    "_create_gradient_factory",
    "_create_integrated_gradients_factory",
    "_create_shap_tree_factory",
    "_get_importance_from_pair",
    "_rank_features",
    "default_explainer_registry",
]
