"""Tests for explainer registry module.

Covers ExplainerRegistry, registrations, factories, adapters, and helper functions.
Uses real backend implementations - no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_ml.explainers.protocol import PredictorProtocol

from covenant_ml.explainers.adapters import (
    _get_importance_from_pair,
    _GradientAdapter,
    _GradientModelWrapper,
    _IntegratedGradientsAdapter,
    _rank_features,
    _ShapTreeAdapter,
)
from covenant_ml.explainers.registry import (
    default_explainer_registry,
)
from tests.explainers._registry_fixtures import (
    _create_cleargbm_prepared,
    _create_tree_predictor,
    _FakePredictorNoGradients,
    _FakePredictorWithGradients,
)


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
