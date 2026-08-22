"""Tests for explainer registry module.

Covers ExplainerRegistry, registrations, factories, adapters, and helper functions.
Uses real backend implementations - no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_ml.explainers.protocol import PredictorProtocol

from covenant_ml.backends.registry import default_registry
from covenant_ml.explainers.adapters import (
    try_extract_native_tree_model,
)
from covenant_ml.explainers.registry import (
    default_explainer_registry,
)
from tests.explainers._registry_fixtures import (
    _create_tree_predictor,
    _FakePredictorNoGradients,
    _FakePredictorWithGradients,
)


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


class TestNativeModelExtraction:
    """Prepared models that wrap a native handle must surrender it to SHAP.

    shap.TreeExplainer reads a model's tree structure and rejects anything it
    does not recognise with "Model type not yet supported by TreeExplainer".
    LightGBM and RandomForest both return wrappers from load(), so shap_tree
    failed for LightGBM and RandomForest was left out of its compatible set
    entirely -- even though SHAP accepts both native handles directly.
    """

    def test_shap_tree_covers_every_tree_backend(self) -> None:
        """Only logreg is excluded, because it is not a tree model."""
        registry = default_explainer_registry()

        without = [
            backend
            for backend in default_registry().list_backends()
            if "shap_tree" not in registry.list_compatible_explainers(backend)
        ]

        assert without == ["logreg"]

    def test_native_extraction_returns_none_for_already_native_models(self) -> None:
        """XGBoost's prepared model is the classifier itself, so nothing to unwrap."""

        class _Native:
            """Stands in for a prepared model that is already native."""

            def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
                """Return a fixed two-column result."""
                n = int(x.shape[0])
                return np.full((n, 2), 0.5, dtype=np.float64)

        assert try_extract_native_tree_model(_Native()) is None

    def test_native_extraction_returns_the_wrapped_model(self) -> None:
        """A prepared model exposing raw_model hands the native handle over."""

        class _Inner:
            """Stands in for a native booster."""

            def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
                """Return one score per row."""
                return np.zeros(int(x.shape[0]), dtype=np.float64)

        inner = _Inner()

        class _Wrapping:
            """Stands in for a prepared model that wraps a native handle."""

            @property
            def raw_model(self) -> _Inner:
                """The wrapped native model."""
                return inner

            def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
                """Return a fixed two-column result."""
                n = int(x.shape[0])
                return np.full((n, 2), 0.5, dtype=np.float64)

        assert try_extract_native_tree_model(_Wrapping()) is inner
