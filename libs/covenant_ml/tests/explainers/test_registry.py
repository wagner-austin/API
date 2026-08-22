"""Tests for explainer registry module.

Covers ExplainerRegistry, registrations, factories, adapters, and helper functions.
Uses real backend implementations - no mocks.
"""

from __future__ import annotations

import pytest
from platform_ml.explainers import FeatureExplainer

from covenant_ml.backends.registry import default_registry
from covenant_ml.explainers.registry import (
    ExplainerRegistration,
    ExplainerRegistry,
    default_explainer_registry,
)
from covenant_ml.explainers.types import SupportedExplainer
from covenant_ml.types import BackendName
from tests.explainers._registry_fixtures import (
    _make_simple_explainer_factory,
)


class TestExplainerRegistration:
    """Tests for ExplainerRegistration class."""

    def test_registration_stores_factory(self) -> None:
        """ExplainerRegistration stores and returns factory."""
        factory = _make_simple_explainer_factory()
        reg = ExplainerRegistration(
            factory=factory,
            compatible_backends=frozenset(["xgboost"]),
            requires_gradients=False,
        )
        # Verify factory is returned correctly
        returned_factory = reg.factory()
        explainer = returned_factory()
        # Verify explainer works by calling its method
        name = explainer.explainer_name()
        assert name == "permutation"

    def test_registration_stores_compatible_backends(self) -> None:
        """ExplainerRegistration stores and returns compatible backends."""
        factory = _make_simple_explainer_factory()
        backends: frozenset[BackendName] = frozenset(["xgboost", "lightgbm"])
        reg = ExplainerRegistration(
            factory=factory,
            compatible_backends=backends,
            requires_gradients=False,
        )
        assert reg.compatible_backends() == backends

    def test_registration_stores_requires_gradients_true(self) -> None:
        """ExplainerRegistration stores requires_gradients=True correctly."""
        factory = _make_simple_explainer_factory()
        reg = ExplainerRegistration(
            factory=factory,
            compatible_backends=frozenset(["mlp"]),
            requires_gradients=True,
        )
        assert reg.requires_gradients() is True

    def test_registration_stores_requires_gradients_false(self) -> None:
        """ExplainerRegistration stores requires_gradients=False correctly."""
        factory = _make_simple_explainer_factory()
        reg = ExplainerRegistration(
            factory=factory,
            compatible_backends=frozenset(["xgboost"]),
            requires_gradients=False,
        )
        assert reg.requires_gradients() is False


class TestExplainerRegistry:
    """Tests for ExplainerRegistry class."""

    def test_registry_starts_empty(self) -> None:
        """New registry has no explainers registered."""
        registry = ExplainerRegistry()
        assert registry.list_explainers() == []

    def test_registry_register_and_list(self) -> None:
        """Registry can register explainer and list it."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()
        reg = ExplainerRegistration(
            factory=factory,
            compatible_backends=frozenset(["xgboost"]),
            requires_gradients=False,
        )
        registry.register("permutation", reg)

        explainers = registry.list_explainers()
        assert explainers == ["permutation"]

    def test_registry_list_sorted_alphabetically(self) -> None:
        """Registry list_explainers returns sorted list."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()

        # Register in non-alphabetical order
        names: list[SupportedExplainer] = ["shap_tree", "gradient", "permutation"]
        for name in names:
            reg = ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["xgboost"]),
                requires_gradients=False,
            )
            registry.register(name, reg)

        explainers = registry.list_explainers()
        assert explainers == ["gradient", "permutation", "shap_tree"]

    def test_registry_list_compatible_explainers_xgboost(self) -> None:
        """Registry filters explainers by xgboost backend."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()

        # Permutation: xgboost and mlp
        registry.register(
            "permutation",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["xgboost", "mlp"]),
                requires_gradients=False,
            ),
        )

        # Gradient: mlp only
        registry.register(
            "gradient",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["mlp"]),
                requires_gradients=True,
            ),
        )

        assert registry.list_compatible_explainers("xgboost") == ["permutation"]

    def test_registry_list_compatible_explainers_mlp(self) -> None:
        """Registry filters explainers by mlp backend."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()

        registry.register(
            "permutation",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["xgboost", "mlp"]),
                requires_gradients=False,
            ),
        )
        registry.register(
            "gradient",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["mlp"]),
                requires_gradients=True,
            ),
        )

        compatible = registry.list_compatible_explainers("mlp")
        assert compatible == ["gradient", "permutation"]

    def test_registry_list_compatible_explainers_empty(self) -> None:
        """Registry returns empty list when no explainers match backend."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()

        registry.register(
            "gradient",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["mlp"]),
                requires_gradients=True,
            ),
        )

        assert registry.list_compatible_explainers("lightgbm") == []

    def test_registry_get_creates_explainer(self) -> None:
        """Registry get() calls factory and returns explainer."""
        call_count = 0

        def counting_factory() -> FeatureExplainer:
            nonlocal call_count
            call_count += 1
            from platform_ml.explainers import PermutationConfig, create_permutation_explainer

            config: PermutationConfig = {"n_repeats": 2, "random_state": 42}
            return create_permutation_explainer(config)

        registry = ExplainerRegistry()
        registry.register(
            "permutation",
            ExplainerRegistration(
                factory=counting_factory,
                compatible_backends=frozenset(["xgboost"]),
                requires_gradients=False,
            ),
        )

        # First get
        result = registry.get("permutation")
        assert call_count == 1
        # Verify it's a working explainer
        name = result.explainer_name()
        assert name == "permutation"

        # Second get creates new instance
        _ = registry.get("permutation")
        assert call_count == 2

    def test_registry_get_raises_for_unknown(self) -> None:
        """Registry get() raises KeyError for unregistered explainer."""
        registry = ExplainerRegistry()

        with pytest.raises(KeyError):
            # Cast to bypass type check - we're testing runtime behavior
            name: SupportedExplainer = "permutation"  # Valid type but not registered
            registry.get(name)

    def test_registry_is_compatible_returns_true(self) -> None:
        """Registry is_compatible returns True for compatible pair."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()
        registry.register(
            "permutation",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["xgboost", "mlp"]),
                requires_gradients=False,
            ),
        )

        assert registry.is_compatible("permutation", "xgboost") is True
        assert registry.is_compatible("permutation", "mlp") is True

    def test_registry_is_compatible_returns_false_for_incompatible(self) -> None:
        """Registry is_compatible returns False for incompatible backend."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()
        registry.register(
            "gradient",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["mlp"]),
                requires_gradients=True,
            ),
        )

        assert registry.is_compatible("gradient", "xgboost") is False

    def test_registry_is_compatible_returns_false_for_unregistered(self) -> None:
        """Registry is_compatible returns False for unregistered explainer."""
        registry = ExplainerRegistry()
        # Use valid type that's not registered
        result = registry.is_compatible("permutation", "xgboost")
        assert result is False

    def test_registry_requires_gradients_true(self) -> None:
        """Registry requires_gradients returns True for gradient explainer."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()
        registry.register(
            "gradient",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["mlp"]),
                requires_gradients=True,
            ),
        )

        assert registry.requires_gradients("gradient") is True

    def test_registry_requires_gradients_false(self) -> None:
        """Registry requires_gradients returns False for permutation explainer."""
        factory = _make_simple_explainer_factory()
        registry = ExplainerRegistry()
        registry.register(
            "permutation",
            ExplainerRegistration(
                factory=factory,
                compatible_backends=frozenset(["xgboost"]),
                requires_gradients=False,
            ),
        )

        assert registry.requires_gradients("permutation") is False

    def test_registry_requires_gradients_raises_for_unknown(self) -> None:
        """Registry requires_gradients raises KeyError for unregistered."""
        registry = ExplainerRegistry()

        with pytest.raises(KeyError):
            registry.requires_gradients("permutation")


class TestDefaultExplainerRegistry:
    """Tests for default_explainer_registry factory."""

    def test_default_registry_has_four_explainers(self) -> None:
        """Default registry has permutation, gradient, integrated_gradients, shap_tree."""
        registry = default_explainer_registry()
        explainers = registry.list_explainers()

        assert len(explainers) == 4
        assert "permutation" in explainers
        assert "gradient" in explainers
        assert "integrated_gradients" in explainers
        assert "shap_tree" in explainers

    def test_default_registry_permutation_compatible_with_xgboost(self) -> None:
        """Permutation explainer is compatible with xgboost."""
        registry = default_explainer_registry()
        assert registry.is_compatible("permutation", "xgboost") is True

    def test_default_registry_permutation_compatible_with_lightgbm(self) -> None:
        """Permutation explainer is compatible with lightgbm."""
        registry = default_explainer_registry()
        assert registry.is_compatible("permutation", "lightgbm") is True

    def test_default_registry_permutation_compatible_with_mlp(self) -> None:
        """Permutation explainer is compatible with mlp."""
        registry = default_explainer_registry()
        assert registry.is_compatible("permutation", "mlp") is True

    def test_default_registry_permutation_compatible_with_lstm(self) -> None:
        """Permutation explainer is compatible with lstm."""
        registry = default_explainer_registry()
        assert registry.is_compatible("permutation", "lstm") is True

    def test_default_registry_gradient_compatible_with_mlp(self) -> None:
        """Gradient explainer is compatible with mlp."""
        registry = default_explainer_registry()
        assert registry.is_compatible("gradient", "mlp") is True

    def test_default_registry_gradient_compatible_with_lstm(self) -> None:
        """Gradient explainer is compatible with lstm."""
        registry = default_explainer_registry()
        assert registry.is_compatible("gradient", "lstm") is True

    def test_default_registry_gradient_not_compatible_with_xgboost(self) -> None:
        """Gradient explainer is not compatible with xgboost."""
        registry = default_explainer_registry()
        assert registry.is_compatible("gradient", "xgboost") is False

    def test_default_registry_gradient_not_compatible_with_lightgbm(self) -> None:
        """Gradient explainer is not compatible with lightgbm."""
        registry = default_explainer_registry()
        assert registry.is_compatible("gradient", "lightgbm") is False

    def test_default_registry_integrated_gradients_compatible_with_mlp(self) -> None:
        """Integrated gradients is compatible with mlp."""
        registry = default_explainer_registry()
        assert registry.is_compatible("integrated_gradients", "mlp") is True

    def test_default_registry_integrated_gradients_compatible_with_lstm(self) -> None:
        """Integrated gradients is compatible with lstm."""
        registry = default_explainer_registry()
        assert registry.is_compatible("integrated_gradients", "lstm") is True

    def test_default_registry_integrated_gradients_not_compatible_with_xgboost(self) -> None:
        """Integrated gradients is not compatible with xgboost."""
        registry = default_explainer_registry()
        assert registry.is_compatible("integrated_gradients", "xgboost") is False

    def test_default_registry_shap_tree_compatible_with_xgboost(self) -> None:
        """SHAP tree is compatible with xgboost."""
        registry = default_explainer_registry()
        assert registry.is_compatible("shap_tree", "xgboost") is True

    def test_default_registry_shap_tree_compatible_with_lightgbm(self) -> None:
        """SHAP tree is compatible with lightgbm."""
        registry = default_explainer_registry()
        assert registry.is_compatible("shap_tree", "lightgbm") is True

    def test_default_registry_shap_tree_not_compatible_with_mlp(self) -> None:
        """SHAP tree is not compatible with mlp."""
        registry = default_explainer_registry()
        assert registry.is_compatible("shap_tree", "mlp") is False

    def test_default_registry_gradient_requires_gradients(self) -> None:
        """Gradient explainer requires gradients."""
        registry = default_explainer_registry()
        assert registry.requires_gradients("gradient") is True

    def test_default_registry_integrated_gradients_requires_gradients(self) -> None:
        """Integrated gradients requires gradients."""
        registry = default_explainer_registry()
        assert registry.requires_gradients("integrated_gradients") is True

    def test_default_registry_permutation_no_gradients(self) -> None:
        """Permutation does not require gradients."""
        registry = default_explainer_registry()
        assert registry.requires_gradients("permutation") is False

    def test_default_registry_shap_tree_no_gradients(self) -> None:
        """SHAP tree does not require gradients."""
        registry = default_explainer_registry()
        assert registry.requires_gradients("shap_tree") is False


class TestDefaultRegistryCoversEveryBackend:
    """Every registered backend has at least one explainer it can use.

    permutation's compatibility set omitted logreg and random_forest while its
    own comment described it as model-agnostic, so both backends had no
    compatible explainer at all: /ml/explain refused every request for them,
    for every explainer, while the API happily accepted the backend name.
    """

    def test_every_backend_has_a_compatible_explainer(self) -> None:
        """No registered backend is left with an empty explainer list."""
        registry = default_explainer_registry()

        empty = [
            backend
            for backend in default_registry().list_backends()
            if not registry.list_compatible_explainers(backend)
        ]

        assert empty == []

    def test_permutation_covers_every_backend(self) -> None:
        """Permutation needs only predict_proba, which every backend has."""
        registry = default_explainer_registry()

        missing = [
            backend
            for backend in default_registry().list_backends()
            if "permutation" not in registry.list_compatible_explainers(backend)
        ]

        assert missing == []

    def test_gradient_explainers_stay_neural_only(self) -> None:
        """Widening permutation must not widen the gradient explainers.

        They need compute_gradients, which the tree and linear backends do
        not have; claiming compatibility would trade a clear refusal for a
        failure inside the explainer.
        """
        registry = default_explainer_registry()

        for backend in ("xgboost", "lightgbm", "cleargbm", "logreg", "random_forest"):
            compatible = registry.list_compatible_explainers(backend)
            assert "gradient" not in compatible
            assert "integrated_gradients" not in compatible
