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

from platform_ml.explainers import (
    PermutationConfig,
    create_regression_permutation_explainer,
)
from platform_ml.explainers.protocol import (
    RegressionFeatureExplainer,
)

from covenant_ml.explainers.regression_adapters import (
    _RegressionGradientAdapter,
    _RegressionIntegratedGradientsAdapter,
    _RegressionShapTreeAdapter,
)
from covenant_ml.types_regression import RegressorBackendName

from .types import SupportedExplainer

# ---------------------------------------------------------------------------
# Gradient model wrapper for regression
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ranking helper (shared by gradient/IG/shap adapters)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regression Gradient Adapter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regression Integrated Gradients Adapter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regression SHAP Tree Adapter
# ---------------------------------------------------------------------------


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
    "default_regression_explainer_registry",
]
