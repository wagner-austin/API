"""Tests for explainer registry module.

Covers ExplainerRegistry, registrations, factories, adapters, and helper functions.
Uses real backend implementations - no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_ml.explainers import FeatureExplainer
from platform_ml.explainers.protocol import PredictorProtocol

from covenant_ml.backends.protocol import PreparedClassifier
from covenant_ml.explainers.registry import (
    ExplainerFactory,
    ExplainerRegistration,
    ExplainerRegistry,
    _get_importance_from_pair,
    _GradientAdapter,
    _GradientModelWrapper,
    _IntegratedGradientsAdapter,
    _rank_features,
    _ShapTreeAdapter,
    default_explainer_registry,
)
from covenant_ml.explainers.types import SupportedExplainer
from covenant_ml.types import BackendName, XGBModelProtocol


class _FakePredictorWithGradients:
    """Fake predictor implementing PredictorProtocol with compute_gradients."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities."""
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.full((n_samples, 2), 0.5, dtype=np.float64)
        return proba

    def compute_gradients(
        self,
        x: NDArray[np.float64],
        target_class: int,
    ) -> NDArray[np.float64]:
        """Return gradients proportional to feature index."""
        n_samples = int(x.shape[0])
        n_features = int(x.shape[1])
        grads: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)
        for f in range(n_features):
            # Gradient magnitude proportional to feature index
            grads[:, f] = float(f + 1) * 0.1
        return grads


class _FakePredictorNoGradients:
    """Fake predictor without compute_gradients method."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return uniform probabilities."""
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.full((n_samples, 2), 0.5, dtype=np.float64)
        return proba


def _create_tree_predictor() -> XGBModelProtocol:
    """Create a real XGBoost model for SHAP tree tests.

    Uses the covenant_ml xgboost backend for training, then loads
    the model directly via xgboost.XGBClassifier for SHAP compatibility.

    Requires SHAP >= 0.50 for XGBoost 3.x compatibility.

    Returns:
        XGBoost trained model implementing XGBModelProtocol.
    """
    import tempfile
    from pathlib import Path

    from covenant_ml.backends.protocol import ClassifierBackend
    from covenant_ml.backends.xgboost import create_xgboost_backend
    from covenant_ml.predictor import load_model
    from covenant_ml.types import TrainConfig

    rng = np.random.default_rng(42)
    x_train = rng.random((100, 4)).astype(np.float64)
    y_train = rng.integers(0, 2, size=100).astype(np.int64)
    feature_names = ["f0", "f1", "f2", "f3"]

    backend: ClassifierBackend = create_xgboost_backend()

    with tempfile.TemporaryDirectory() as tmp_dir:
        config: TrainConfig = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_state": 42,
            "early_stopping_rounds": 3,
            "device": "cpu",
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }

        outcome = backend.train(
            x_features=x_train,
            y_labels=y_train,
            feature_names=feature_names,
            config=config,
            output_dir=Path(tmp_dir),
            progress=None,
        )

        # Load model directly using XGBClassifier (not backend.load())
        # This creates a real XGBoost model that SHAP TreeExplainer can use
        return load_model(outcome["model_path"])


def _make_simple_explainer_factory() -> ExplainerFactory:
    """Create a factory that returns a simple explainer.

    Returns:
        Factory function that creates permutation explainer.
    """
    from platform_ml.explainers import PermutationConfig, create_permutation_explainer

    def factory() -> FeatureExplainer:
        config: PermutationConfig = {"n_repeats": 2, "random_state": 42}
        return create_permutation_explainer(config)

    return factory


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


class TestGradientModelWrapper:
    """Tests for _GradientModelWrapper class."""

    def test_wrapper_accepts_model_with_gradients(self) -> None:
        """Wrapper accepts model with compute_gradients method."""
        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model
        wrapper = _GradientModelWrapper(predictor)

        # Verify wrapper is functional by calling compute_gradients
        x = np.random.randn(2, 3).astype(np.float64)
        grads = wrapper.compute_gradients(x, target_class=1)
        assert grads.shape == (2, 3)

    def test_wrapper_raises_for_model_without_gradients(self) -> None:
        """Wrapper raises AttributeError for model without compute_gradients."""
        model = _FakePredictorNoGradients()
        predictor: PredictorProtocol = model

        with pytest.raises(AttributeError, match="must have compute_gradients"):
            _GradientModelWrapper(predictor)

    def test_wrapper_compute_gradients_returns_correct_shape(self) -> None:
        """Wrapper compute_gradients returns array with correct shape."""
        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model
        wrapper = _GradientModelWrapper(predictor)

        x = np.random.randn(5, 4).astype(np.float64)
        grads = wrapper.compute_gradients(x, target_class=1)

        assert grads.shape == (5, 4)
        assert grads.dtype == np.float64


class TestRankFeaturesHelper:
    """Tests for _rank_features helper function."""

    def test_rank_features_sorts_by_importance_descending(self) -> None:
        """_rank_features sorts features by importance (highest first)."""
        feature_names = ["a", "b", "c"]
        importances: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        importances[0] = 0.1
        importances[1] = 0.5
        importances[2] = 0.3

        ranked = _rank_features(feature_names, importances)

        assert len(ranked) == 3
        assert ranked[0]["name"] == "b"
        assert ranked[0]["importance"] == 0.5
        assert ranked[0]["rank"] == 1
        assert ranked[1]["name"] == "c"
        assert ranked[1]["importance"] == 0.3
        assert ranked[1]["rank"] == 2
        assert ranked[2]["name"] == "a"
        assert ranked[2]["importance"] == 0.1
        assert ranked[2]["rank"] == 3

    def test_rank_features_handles_equal_importances(self) -> None:
        """_rank_features handles features with equal importance."""
        feature_names = ["x", "y", "z"]
        importances: NDArray[np.float64] = np.full(3, 0.5, dtype=np.float64)

        ranked = _rank_features(feature_names, importances)

        assert len(ranked) == 3
        # All have same importance, order may vary but ranks should be 1, 2, 3
        ranks = [r["rank"] for r in ranked]
        assert sorted(ranks) == [1, 2, 3]

    def test_rank_features_handles_single_feature(self) -> None:
        """_rank_features handles single feature."""
        feature_names = ["only"]
        importances: NDArray[np.float64] = np.full(1, 1.0, dtype=np.float64)

        ranked = _rank_features(feature_names, importances)

        assert len(ranked) == 1
        assert ranked[0]["name"] == "only"
        assert ranked[0]["rank"] == 1


class TestGetImportanceFromPairHelper:
    """Tests for _get_importance_from_pair helper function."""

    def test_extracts_importance_value(self) -> None:
        """_get_importance_from_pair extracts second element of tuple."""
        pair: tuple[int, float] = (3, 0.75)
        importance = _get_importance_from_pair(pair)
        assert importance == 0.75

    def test_handles_negative_importance(self) -> None:
        """_get_importance_from_pair handles negative values."""
        pair: tuple[int, float] = (0, -0.5)
        importance = _get_importance_from_pair(pair)
        assert importance == -0.5


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


class TestPermutationExplainerFromRegistry:
    """Tests for permutation explainer created from registry."""

    def test_permutation_explainer_computes_importance(self) -> None:
        """Permutation explainer computes feature importance scores."""
        registry = default_explainer_registry()
        explainer = registry.get("permutation")

        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model
        x = np.random.randn(20, 4).astype(np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = explainer.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        assert len(scores) == 4
        # Check all required fields present with actual values
        for score in scores:
            assert score["name"] in feature_names
            # Importance is a float - verify by arithmetic operation
            imp_float: float = score["importance"]
            assert imp_float == imp_float  # NaN check
            assert 1 <= score["rank"] <= 4


class TestGradientExplainerFromRegistry:
    """Tests for gradient explainer created from registry."""

    def test_gradient_explainer_computes_importance(self) -> None:
        """Gradient explainer computes feature importance using gradients."""
        registry = default_explainer_registry()
        explainer = registry.get("gradient")

        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model
        # Use deterministic input (ones) so gradient ranking is preserved
        # Gradient adapter multiplies grads by input, so ones preserve ranking
        x: NDArray[np.float64] = np.ones((10, 4), dtype=np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = explainer.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        assert len(scores) == 4
        # Higher feature index = higher gradient in our fake model
        # So f3 should be ranked highest
        assert scores[0]["name"] == "f3"
        assert scores[0]["rank"] == 1

    def test_gradient_explainer_raises_without_gradients(self) -> None:
        """Gradient explainer raises for model without compute_gradients."""
        registry = default_explainer_registry()
        explainer = registry.get("gradient")

        model = _FakePredictorNoGradients()
        predictor: PredictorProtocol = model
        x = np.random.randn(10, 4).astype(np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        with pytest.raises(AttributeError, match="must have compute_gradients"):
            explainer.compute_importance(
                model=predictor,
                x_data=x,
                feature_names=feature_names,
                target_class=1,
            )

    def test_gradient_explainer_capabilities_requires_gradients(self) -> None:
        """Gradient explainer requires gradients."""
        registry = default_explainer_registry()
        explainer = registry.get("gradient")

        caps = explainer.capabilities()
        assert caps["requires_gradients"] is True

    def test_gradient_explainer_capabilities_no_background_data(self) -> None:
        """Gradient explainer does not require background data."""
        registry = default_explainer_registry()
        explainer = registry.get("gradient")

        caps = explainer.capabilities()
        assert caps["requires_background_data"] is False

    def test_gradient_explainer_capabilities_low_cost(self) -> None:
        """Gradient explainer has low computational cost."""
        registry = default_explainer_registry()
        explainer = registry.get("gradient")

        caps = explainer.capabilities()
        assert caps["computational_cost"] == "low"

    def test_gradient_explainer_name(self) -> None:
        """Gradient explainer reports correct name."""
        registry = default_explainer_registry()
        explainer = registry.get("gradient")

        name = explainer.explainer_name()
        assert name == "gradient"


class TestIntegratedGradientsExplainerFromRegistry:
    """Tests for integrated gradients explainer created from registry."""

    def test_integrated_gradients_computes_importance(self) -> None:
        """Integrated gradients explainer computes feature importance."""
        registry = default_explainer_registry()
        explainer = registry.get("integrated_gradients")

        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model
        x = np.random.randn(5, 4).astype(np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = explainer.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        assert len(scores) == 4
        for score in scores:
            assert score["name"] in feature_names
            # Importance is a float - verify by arithmetic operation
            imp_float: float = score["importance"]
            assert imp_float == imp_float  # NaN check
            assert 1 <= score["rank"] <= 4

    def test_integrated_gradients_raises_without_gradients(self) -> None:
        """Integrated gradients raises for model without compute_gradients."""
        registry = default_explainer_registry()
        explainer = registry.get("integrated_gradients")

        model = _FakePredictorNoGradients()
        predictor: PredictorProtocol = model
        x = np.random.randn(5, 4).astype(np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        with pytest.raises(AttributeError, match="must have compute_gradients"):
            explainer.compute_importance(
                model=predictor,
                x_data=x,
                feature_names=feature_names,
                target_class=1,
            )

    def test_integrated_gradients_capabilities_requires_gradients(self) -> None:
        """Integrated gradients requires gradients."""
        registry = default_explainer_registry()
        explainer = registry.get("integrated_gradients")

        caps = explainer.capabilities()
        assert caps["requires_gradients"] is True

    def test_integrated_gradients_capabilities_high_cost(self) -> None:
        """Integrated gradients has high computational cost."""
        registry = default_explainer_registry()
        explainer = registry.get("integrated_gradients")

        caps = explainer.capabilities()
        assert caps["computational_cost"] == "high"

    def test_integrated_gradients_name(self) -> None:
        """Integrated gradients explainer reports correct name."""
        registry = default_explainer_registry()
        explainer = registry.get("integrated_gradients")

        name = explainer.explainer_name()
        assert name == "integrated_gradients"


class TestShapTreeExplainerFromRegistry:
    """Tests for SHAP tree explainer created from registry."""

    def test_shap_tree_computes_importance(self) -> None:
        """SHAP tree explainer computes feature importance using SHAP."""
        registry = default_explainer_registry()
        explainer = registry.get("shap_tree")

        predictor = _create_tree_predictor()
        x = np.random.randn(10, 4).astype(np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = explainer.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        assert len(scores) == 4
        for score in scores:
            assert score["name"] in feature_names
            # Importance should be non-negative (we use abs)
            assert score["importance"] >= 0.0
            assert 1 <= score["rank"] <= 4

    def test_shap_tree_capabilities_no_gradients(self) -> None:
        """SHAP tree does not require gradients."""
        registry = default_explainer_registry()
        explainer = registry.get("shap_tree")

        caps = explainer.capabilities()
        assert caps["requires_gradients"] is False

    def test_shap_tree_capabilities_medium_cost(self) -> None:
        """SHAP tree has medium computational cost."""
        registry = default_explainer_registry()
        explainer = registry.get("shap_tree")

        caps = explainer.capabilities()
        assert caps["computational_cost"] == "medium"

    def test_shap_tree_explainer_name(self) -> None:
        """SHAP tree explainer returns placeholder name."""
        registry = default_explainer_registry()
        explainer = registry.get("shap_tree")

        # Note: explainer_name returns "permutation" as placeholder
        # since ExplainerName type doesn't include "shap_tree"
        name = explainer.explainer_name()
        assert name == "permutation"


class TestGradientAdapterConfigurations:
    """Tests for _GradientAdapter with different configurations."""

    def test_gradient_adapter_without_multiply_by_input(self) -> None:
        """GradientAdapter computes importance without multiplying by input."""
        adapter = _GradientAdapter(multiply_by_input=False, absolute_value=True)
        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model

        # Create input data with specific values
        x: NDArray[np.float64] = np.ones((5, 4), dtype=np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = adapter.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        # Verify computation works
        assert len(scores) == 4
        # With multiply_by_input=False, gradients are used directly
        # Our fake model returns gradients proportional to feature index
        assert scores[0]["name"] == "f3"  # Highest gradient
        assert scores[0]["rank"] == 1

    def test_gradient_adapter_without_absolute_value(self) -> None:
        """GradientAdapter computes importance without absolute value."""
        adapter = _GradientAdapter(multiply_by_input=True, absolute_value=False)
        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model

        x: NDArray[np.float64] = np.ones((5, 4), dtype=np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = adapter.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        # Verify computation works
        assert len(scores) == 4
        # Check scores are computed (may be negative without abs)
        for score in scores:
            assert score["name"] in feature_names
            assert 1 <= score["rank"] <= 4

    def test_gradient_adapter_both_disabled(self) -> None:
        """GradientAdapter with both multiply_by_input and absolute_value disabled."""
        adapter = _GradientAdapter(multiply_by_input=False, absolute_value=False)
        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model

        x: NDArray[np.float64] = np.ones((5, 4), dtype=np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = adapter.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        # Verify computation completes and returns valid structure
        assert len(scores) == 4
        for score in scores:
            assert "name" in score
            assert "importance" in score
            assert "rank" in score


class TestIntegratedGradientsAdapterConfigurations:
    """Tests for _IntegratedGradientsAdapter with different configurations."""

    def test_integrated_gradients_adapter_mean_baseline(self) -> None:
        """IntegratedGradientsAdapter uses mean baseline when configured."""
        adapter = _IntegratedGradientsAdapter(n_steps=10, baseline_mode="mean")
        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model

        # Create input data where mean differs from zeros
        x: NDArray[np.float64] = np.zeros((5, 4), dtype=np.float64)
        x[0, :] = 1.0
        x[1, :] = 2.0
        x[2, :] = 3.0
        x[3, :] = 4.0
        x[4, :] = 5.0
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = adapter.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        # Verify computation works with mean baseline
        assert len(scores) == 4
        for score in scores:
            assert score["name"] in feature_names
            # Importance should be non-negative (abs is always applied)
            assert score["importance"] >= 0.0
            assert 1 <= score["rank"] <= 4

    def test_integrated_gradients_adapter_zeros_baseline(self) -> None:
        """IntegratedGradientsAdapter uses zeros baseline when configured."""
        adapter = _IntegratedGradientsAdapter(n_steps=5, baseline_mode="zeros")
        model = _FakePredictorWithGradients()
        predictor: PredictorProtocol = model

        x: NDArray[np.float64] = np.ones((3, 4), dtype=np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = adapter.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        # Verify computation works with zeros baseline
        assert len(scores) == 4

    def test_integrated_gradients_adapter_explainer_name(self) -> None:
        """IntegratedGradientsAdapter returns correct explainer name."""
        adapter = _IntegratedGradientsAdapter(n_steps=10, baseline_mode="zeros")
        name = adapter.explainer_name()
        assert name == "integrated_gradients"

    def test_integrated_gradients_adapter_capabilities(self) -> None:
        """IntegratedGradientsAdapter returns correct capabilities."""
        adapter = _IntegratedGradientsAdapter(n_steps=10, baseline_mode="zeros")
        caps = adapter.capabilities()

        assert caps["requires_gradients"] is True
        assert caps["requires_background_data"] is False
        assert caps["computational_cost"] == "high"


class TestGradientAdapterMetadata:
    """Tests for _GradientAdapter metadata methods."""

    def test_gradient_adapter_explainer_name(self) -> None:
        """GradientAdapter returns correct explainer name."""
        adapter = _GradientAdapter(multiply_by_input=True, absolute_value=True)
        name = adapter.explainer_name()
        assert name == "gradient"

    def test_gradient_adapter_capabilities(self) -> None:
        """GradientAdapter returns correct capabilities."""
        adapter = _GradientAdapter(multiply_by_input=True, absolute_value=True)
        caps = adapter.capabilities()

        assert caps["requires_gradients"] is True
        assert caps["requires_background_data"] is False
        assert caps["computational_cost"] == "low"


class TestShapTreeAdapterDirect:
    """Direct tests for _ShapTreeAdapter class."""

    def test_shap_tree_adapter_explainer_name(self) -> None:
        """ShapTreeAdapter returns placeholder explainer name."""
        adapter = _ShapTreeAdapter()
        name = adapter.explainer_name()
        # Returns "permutation" as placeholder since ExplainerName doesn't include shap_tree
        assert name == "permutation"

    def test_shap_tree_adapter_capabilities(self) -> None:
        """ShapTreeAdapter returns correct capabilities."""
        adapter = _ShapTreeAdapter()
        caps = adapter.capabilities()

        assert caps["requires_gradients"] is False
        assert caps["requires_background_data"] is False
        assert caps["computational_cost"] == "medium"

    def test_shap_tree_adapter_computes_importance(self) -> None:
        """ShapTreeAdapter computes feature importance correctly."""
        adapter = _ShapTreeAdapter()
        predictor = _create_tree_predictor()

        x: NDArray[np.float64] = np.random.randn(5, 4).astype(np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = adapter.compute_importance(
            model=predictor,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        assert len(scores) == 4
        for score in scores:
            assert score["name"] in feature_names
            assert score["importance"] >= 0.0
            assert 1 <= score["rank"] <= 4


# =============================================================================
# ClearGBM integration tests
# =============================================================================


def _create_cleargbm_prepared() -> PreparedClassifier:
    """Create a ClearGBM prepared classifier for tests.

    Uses the ClearGBM backend to create a real _ClearGBMPrepared instance
    that will be recognized by try_extract_cleargbm_model.

    Returns:
        PreparedClassifier wrapping a ClearGBM model.
    """
    import tempfile
    from pathlib import Path

    from cleargbm.ensemble import train_gradient_boosting
    from cleargbm.types import GradientBoostingConfig, encode_gradient_boosting_model
    from platform_core.json_utils import dump_json_str

    from covenant_ml.backends.cleargbm import ClearGBMBackend

    rng = np.random.default_rng(42)
    x_train = rng.random((100, 4)).astype(np.float64)
    y_train = rng.integers(0, 2, size=100).astype(np.int64)

    # Convert to tuples for cleargbm (explicit loops to avoid Any from generators)
    rows: list[tuple[float, ...]] = []
    n_rows = int(x_train.shape[0])
    for i in range(n_rows):
        row_arr: NDArray[np.float64] = x_train[i, :]
        row_list: list[float] = []
        n_cols = int(row_arr.shape[0])
        for j in range(n_cols):
            val: float = float(row_arr.flat[j].item())
            row_list.append(val)
        rows.append(tuple(row_list))
    x_tuple: tuple[tuple[float, ...], ...] = tuple(rows)

    labels: list[int] = []
    n_labels = int(y_train.shape[0])
    for i in range(n_labels):
        label_val: int = int(y_train.flat[i].item())
        labels.append(label_val)
    y_tuple: tuple[int, ...] = tuple(labels)

    feature_names: tuple[str, ...] = ("f0", "f1", "f2", "f3")

    config = GradientBoostingConfig(
        n_estimators=5,
        max_depth=3,
        learning_rate=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        max_bins=64,
        subsample=1.0,
        random_state=42,
        track_contributions=False,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=1,
    )

    gbm_model = train_gradient_boosting(
        x_train=x_tuple,
        y_train=y_tuple,
        x_val=None,
        y_val=None,
        config=config,
        feature_names=feature_names,
        progress_callback=None,
    )

    # Use the backend to save/load - this creates a real _ClearGBMPrepared
    backend = ClearGBMBackend()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.json"

        # Save model directly to JSON
        encoded = encode_gradient_boosting_model(gbm_model)
        json_str = dump_json_str(encoded, indent=2)
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(json_str)

        # Load via backend to get the real _ClearGBMPrepared type
        return backend.load(path=str(model_path))


class TestShapTreeAdapterWithClearGBM:
    """Tests for _ShapTreeAdapter with ClearGBM models."""

    def test_adapter_computes_importance_for_cleargbm(self) -> None:
        """ShapTreeAdapter computes importance for ClearGBM model."""
        adapter = _ShapTreeAdapter()
        prepared = _create_cleargbm_prepared()

        x: NDArray[np.float64] = np.random.randn(5, 4).astype(np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = adapter.compute_importance(
            model=prepared,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        assert len(scores) == 4
        for score in scores:
            assert score["name"] in feature_names
            assert score["importance"] >= 0.0
            assert 1 <= score["rank"] <= 4

    def test_registry_shap_tree_works_with_cleargbm(self) -> None:
        """SHAP tree explainer from registry works with ClearGBM model."""
        registry = default_explainer_registry()
        explainer = registry.get("shap_tree")
        prepared = _create_cleargbm_prepared()

        x: NDArray[np.float64] = np.random.randn(3, 4).astype(np.float64)
        feature_names = ["f0", "f1", "f2", "f3"]

        scores = explainer.compute_importance(
            model=prepared,
            x_data=x,
            feature_names=feature_names,
            target_class=1,
        )

        assert len(scores) == 4
        for score in scores:
            assert score["name"] in feature_names
            assert score["importance"] >= 0.0
            assert 1 <= score["rank"] <= 4
