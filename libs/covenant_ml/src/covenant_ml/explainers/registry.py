"""Explainer registry with backend compatibility mapping.

Provides a registry of feature importance explainers with explicit
backend compatibility information. Explainers are registered with
factory functions and metadata about their requirements.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Protocol

from platform_ml.explainers import (
    FeatureExplainer,
    PermutationConfig,
    create_permutation_explainer,
)

from covenant_ml.explainers.adapters import (
    _GradientAdapter,
    _IntegratedGradientsAdapter,
    _ShapTreeAdapter,
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


# ---------------------------------------------------------------------------
# Helper functions for ranking to avoid Any types
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Local TypedDict for SHAP explanations (not exported from platform_ml)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Adapter classes for explainer implementations
# ---------------------------------------------------------------------------


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

    Returns adapter that wraps ShapTreeWrapper (for XGBoost/LightGBM) or
    ClearGBMShapWrapper (for ClearGBM) to match FeatureExplainer protocol.
    Model type is auto-detected at runtime.
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

    # Permutation needs only predict_proba, so it works with every backend.
    # logreg and random_forest were missing from this set, which left them
    # with no compatible explainer at all: /ml/explain refused every request
    # for them, for every explainer, while the API accepted the backend.
    reg.register(
        "permutation",
        ExplainerRegistration(
            factory=_create_permutation_factory(),
            compatible_backends=frozenset(
                ["xgboost", "lightgbm", "mlp", "lstm", "cleargbm", "logreg", "random_forest"]
            ),
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

    # SHAP Tree: every tree-based backend. random_forest was omitted, though
    # shap.TreeExplainer accepts sklearn ensembles directly -- what it rejects
    # is the prepared wrapper, which is now unwrapped before it is handed over.
    # logreg is excluded because it is not a tree model at all.
    reg.register(
        "shap_tree",
        ExplainerRegistration(
            factory=_create_shap_tree_factory(),
            compatible_backends=frozenset(["xgboost", "lightgbm", "cleargbm", "random_forest"]),
            requires_gradients=False,
        ),
    )

    return reg


__all__ = [
    "ExplainerFactory",
    "ExplainerRegistration",
    "ExplainerRegistry",
    "_create_gradient_factory",
    "_create_integrated_gradients_factory",
    "_create_shap_tree_factory",
    "default_explainer_registry",
]
