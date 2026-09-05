"""Tests for regression explainer registry module.

Covers RegressionExplainerRegistry, registrations, factories, adapters,
and helper functions. Uses real implementations - no mocks.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray
from platform_ml.explainers.protocol import (
    RegressionFeatureExplainer,
    RegressorPredictorProtocol,
)

from covenant_ml.explainers.regression_adapters import (
    _RegressionGradientAdapter,
    _RegressionGradientModelWrapper,
    _RegressionIntegratedGradientsAdapter,
)
from covenant_ml.explainers.regression_registry import (
    RegressionExplainerRegistration,
    RegressionExplainerRegistry,
    default_regression_explainer_registry,
)
from covenant_ml.types_regression import RegressorBackendName


class _TrainableRegressorProto(Protocol):
    """Regressor that supports both fit and predict.

    Used for typed dynamic imports of XGBRegressor in tests.
    Structurally compatible with RegressorPredictorProtocol (has predict).
    """

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> _TrainableRegressorProto: ...

    def predict(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


def _make_x(rows: list[list[float]]) -> NDArray[np.float64]:
    """Create 2D float64 array without list[Any] mypy issues.

    Args:
        rows: List of rows.

    Returns:
        2D numpy array.
    """
    n_rows = len(rows)
    n_cols = len(rows[0]) if n_rows > 0 else 0
    result: NDArray[np.float64] = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            result[i, j] = v
    return result


class _FakeRegressorWithGradients:
    """Fake regressor implementing RegressorPredictorProtocol + gradients."""

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return sum of features as prediction.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Predicted values with shape (n_samples,).
        """
        n_samples = int(x.shape[0])
        result: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            total = 0.0
            for j in range(int(x.shape[1])):
                total += float(x.flat[i * int(x.shape[1]) + j].item())
            result[i] = total
        return result

    def compute_regression_gradients(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return gradients proportional to feature index.

        Args:
            x: Input features with shape (n_samples, n_features).

        Returns:
            Gradients with shape (n_samples, n_features).
        """
        n_samples = int(x.shape[0])
        n_features = int(x.shape[1])
        grads: NDArray[np.float64] = np.zeros((n_samples, n_features), dtype=np.float64)
        for f in range(n_features):
            grads[:, f] = float(f + 1) * 0.1
        return grads


class _FakeRegressorNoGradients:
    """Fake regressor without compute_regression_gradients method."""

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return constant predictions.

        Args:
            x: Input features.

        Returns:
            Array of 1.0 values.
        """
        n_samples = int(x.shape[0])
        return np.ones(n_samples, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestRegressionGradientModelWrapper:
    """Tests for _RegressionGradientModelWrapper."""

    def test_delegates_to_model(self) -> None:
        """Wrapper delegates compute_regression_gradients to model."""
        model = _FakeRegressorWithGradients()
        wrapper = _RegressionGradientModelWrapper(model)
        x = _make_x([[1.0, 2.0]])
        grads = wrapper.compute_regression_gradients(x)

        assert grads.shape == (1, 2)
        assert float(grads.flat[0].item()) == pytest.approx(0.1)
        assert float(grads.flat[1].item()) == pytest.approx(0.2)

    def test_raises_without_method(self) -> None:
        """Wrapper raises AttributeError if model lacks method."""
        model = _FakeRegressorNoGradients()
        wrapper = _RegressionGradientModelWrapper(model)
        x = _make_x([[1.0, 2.0]])
        with pytest.raises(AttributeError):
            wrapper.compute_regression_gradients(x)


# ---------------------------------------------------------------------------
# Gradient adapter tests
# ---------------------------------------------------------------------------


class TestRegressionGradientAdapter:
    """Tests for _RegressionGradientAdapter."""

    def test_explainer_name(self) -> None:
        """Returns 'gradient'."""
        adapter = _RegressionGradientAdapter(multiply_by_input=True, absolute_value=True)
        assert adapter.explainer_name() == "gradient"

    def test_capabilities(self) -> None:
        """Capabilities require gradients."""
        adapter = _RegressionGradientAdapter(multiply_by_input=True, absolute_value=True)
        caps = adapter.capabilities()
        assert caps["requires_gradients"] is True
        assert caps["computational_cost"] == "low"

    def test_compute_importance(self) -> None:
        """Gradient adapter produces ranked importance scores."""
        adapter = _RegressionGradientAdapter(multiply_by_input=True, absolute_value=True)
        model: RegressorPredictorProtocol = _FakeRegressorWithGradients()
        x = _make_x([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        importance = adapter.compute_importance(
            model=model, x_data=x, feature_names=["a", "b", "c"]
        )

        assert len(importance) == 3
        # Feature 2 (index=2, grad=0.3) should rank highest
        assert importance[0]["rank"] == 1

    def test_without_multiply_by_input(self) -> None:
        """Gradient adapter works without multiply_by_input."""
        adapter = _RegressionGradientAdapter(multiply_by_input=False, absolute_value=True)
        model: RegressorPredictorProtocol = _FakeRegressorWithGradients()
        x = _make_x([[1.0, 2.0], [3.0, 4.0]])

        importance = adapter.compute_importance(model=model, x_data=x, feature_names=["a", "b"])

        assert len(importance) == 2

    def test_without_absolute_value(self) -> None:
        """Gradient adapter works without absolute_value."""
        adapter = _RegressionGradientAdapter(multiply_by_input=True, absolute_value=False)
        model: RegressorPredictorProtocol = _FakeRegressorWithGradients()
        x = _make_x([[1.0, 2.0], [3.0, 4.0]])

        importance = adapter.compute_importance(model=model, x_data=x, feature_names=["a", "b"])

        assert len(importance) == 2


# ---------------------------------------------------------------------------
# Integrated gradients adapter tests
# ---------------------------------------------------------------------------


class TestRegressionIntegratedGradientsAdapter:
    """Tests for _RegressionIntegratedGradientsAdapter."""

    def test_explainer_name(self) -> None:
        """Returns 'integrated_gradients'."""
        adapter = _RegressionIntegratedGradientsAdapter(n_steps=10, baseline_mode="zeros")
        assert adapter.explainer_name() == "integrated_gradients"

    def test_capabilities(self) -> None:
        """Capabilities require gradients, high cost."""
        adapter = _RegressionIntegratedGradientsAdapter(n_steps=10, baseline_mode="zeros")
        caps = adapter.capabilities()
        assert caps["requires_gradients"] is True
        assert caps["computational_cost"] == "high"

    def test_compute_importance_zeros_baseline(self) -> None:
        """IG with zeros baseline produces importance scores."""
        adapter = _RegressionIntegratedGradientsAdapter(n_steps=10, baseline_mode="zeros")
        model: RegressorPredictorProtocol = _FakeRegressorWithGradients()
        x = _make_x([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        importance = adapter.compute_importance(
            model=model, x_data=x, feature_names=["a", "b", "c"]
        )

        assert len(importance) == 3
        assert importance[0]["rank"] == 1

    def test_compute_importance_mean_baseline(self) -> None:
        """IG with mean baseline produces importance scores."""
        adapter = _RegressionIntegratedGradientsAdapter(n_steps=5, baseline_mode="mean")
        model: RegressorPredictorProtocol = _FakeRegressorWithGradients()
        x = _make_x([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        importance = adapter.compute_importance(
            model=model, x_data=x, feature_names=["a", "b", "c"]
        )

        assert len(importance) == 3


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegressionExplainerRegistration:
    """Tests for RegressionExplainerRegistration."""

    def test_factory(self) -> None:
        """Factory returns the provided factory function."""

        def _factory() -> RegressionFeatureExplainer:
            return _RegressionGradientAdapter(multiply_by_input=True, absolute_value=True)

        reg = RegressionExplainerRegistration(
            factory=_factory,
            compatible_backends=frozenset(["mlp_reg", "lstm_reg"]),
            requires_gradients=True,
        )
        assert reg.factory() is _factory

    def test_compatible_backends(self) -> None:
        """Returns the registered compatible backends."""

        def _factory() -> RegressionFeatureExplainer:
            return _RegressionGradientAdapter(multiply_by_input=True, absolute_value=True)

        backends: frozenset[RegressorBackendName] = frozenset(["mlp_reg", "lstm_reg"])
        reg = RegressionExplainerRegistration(
            factory=_factory,
            compatible_backends=backends,
            requires_gradients=True,
        )
        assert reg.compatible_backends() == backends

    def test_requires_gradients(self) -> None:
        """Returns the requires_gradients flag."""

        def _factory() -> RegressionFeatureExplainer:
            return _RegressionGradientAdapter(multiply_by_input=True, absolute_value=True)

        reg = RegressionExplainerRegistration(
            factory=_factory,
            compatible_backends=frozenset(["mlp_reg"]),
            requires_gradients=True,
        )
        assert reg.requires_gradients() is True


class TestRegressionExplainerRegistry:
    """Tests for RegressionExplainerRegistry."""

    def test_register_and_list(self) -> None:
        """Register explainers and list them."""
        registry = RegressionExplainerRegistry()

        def _factory() -> RegressionFeatureExplainer:
            return _RegressionGradientAdapter(multiply_by_input=True, absolute_value=True)

        registry.register(
            "gradient",
            RegressionExplainerRegistration(
                factory=_factory,
                compatible_backends=frozenset(["mlp_reg"]),
                requires_gradients=True,
            ),
        )

        explainers = registry.list_explainers()
        assert explainers == ["gradient"]

    def test_list_compatible(self) -> None:
        """List compatible explainers for a given backend."""
        registry = default_regression_explainer_registry()

        xgb_compatible = registry.list_compatible_explainers("xgboost_reg")
        assert "permutation" in xgb_compatible
        assert "shap_tree" in xgb_compatible
        assert "gradient" not in xgb_compatible

        mlp_compatible = registry.list_compatible_explainers("mlp_reg")
        assert "permutation" in mlp_compatible
        assert "gradient" in mlp_compatible
        assert "integrated_gradients" in mlp_compatible
        assert "shap_tree" not in mlp_compatible

    def test_get_creates_instance(self) -> None:
        """Get creates a new explainer instance."""
        registry = default_regression_explainer_registry()
        explainer = registry.get("permutation")
        assert explainer.explainer_name() == "permutation"

    def test_get_unknown_raises(self) -> None:
        """Get raises KeyError for unknown explainer."""
        registry = RegressionExplainerRegistry()
        with pytest.raises(KeyError):
            registry.get("permutation")

    def test_is_compatible_true(self) -> None:
        """is_compatible returns True for compatible pair."""
        registry = default_regression_explainer_registry()
        assert registry.is_compatible("permutation", "xgboost_reg") is True

    def test_is_compatible_false(self) -> None:
        """is_compatible returns False for incompatible pair."""
        registry = default_regression_explainer_registry()
        assert registry.is_compatible("shap_tree", "mlp_reg") is False

    def test_is_compatible_unknown_explainer(self) -> None:
        """is_compatible returns False for unregistered explainer."""
        registry = RegressionExplainerRegistry()
        assert registry.is_compatible("permutation", "xgboost_reg") is False


class TestDefaultRegressionExplainerRegistry:
    """Tests for default_regression_explainer_registry factory."""

    def test_has_all_four_explainers(self) -> None:
        """Default registry has all four explainer types."""
        registry = default_regression_explainer_registry()
        explainers = registry.list_explainers()
        assert "gradient" in explainers
        assert "integrated_gradients" in explainers
        assert "permutation" in explainers
        assert "shap_tree" in explainers

    def test_lightgbm_reg_compatible(self) -> None:
        """LightGBM regressor has permutation + shap_tree."""
        registry = default_regression_explainer_registry()
        compatible = registry.list_compatible_explainers("lightgbm_reg")
        assert sorted(compatible) == ["permutation", "shap_tree"]

    def test_lstm_reg_compatible(self) -> None:
        """LSTM regressor has gradient + IG + permutation."""
        registry = default_regression_explainer_registry()
        compatible = registry.list_compatible_explainers("lstm_reg")
        assert sorted(compatible) == ["gradient", "integrated_gradients", "permutation"]

    def test_get_gradient_creates_instance(self) -> None:
        """Get gradient creates a RegressionGradientAdapter instance."""
        registry = default_regression_explainer_registry()
        explainer = registry.get("gradient")
        assert explainer.explainer_name() == "gradient"

    def test_get_integrated_gradients_creates_instance(self) -> None:
        """Get integrated_gradients creates the right adapter."""
        registry = default_regression_explainer_registry()
        explainer = registry.get("integrated_gradients")
        assert explainer.explainer_name() == "integrated_gradients"

    def test_get_shap_tree_creates_instance(self) -> None:
        """Get shap_tree creates a RegressionShapTreeAdapter instance."""
        registry = default_regression_explainer_registry()
        explainer = registry.get("shap_tree")
        caps = explainer.capabilities()
        assert caps["requires_gradients"] is False


class TestRegressionShapTreeAdapter:
    """Tests for _RegressionShapTreeAdapter."""

    def test_explainer_name(self) -> None:
        """Returns 'shap_tree'."""
        from covenant_ml.explainers.regression_adapters import _RegressionShapTreeAdapter

        adapter = _RegressionShapTreeAdapter()
        assert adapter.explainer_name() == "shap_tree"

    def test_capabilities(self) -> None:
        """Capabilities do not require gradients."""
        from covenant_ml.explainers.regression_adapters import _RegressionShapTreeAdapter

        adapter = _RegressionShapTreeAdapter()
        caps = adapter.capabilities()
        assert caps["requires_gradients"] is False
        assert caps["computational_cost"] == "medium"

    def test_compute_importance_with_raw_xgboost(self) -> None:
        """SHAP tree adapter works with a raw XGBRegressor."""
        from covenant_ml.explainers.regression_adapters import _RegressionShapTreeAdapter

        xgb_mod = __import__("xgboost")
        regressor: _TrainableRegressorProto = xgb_mod.XGBRegressor(
            n_estimators=5,
            max_depth=2,
            random_state=42,
        )

        x_train = _make_x([[1.0, 2.0], [3.0, 1.0], [5.0, 0.5], [2.0, 4.0]])
        y_train: NDArray[np.float64] = np.arange(1.0, 5.0, dtype=np.float64)
        regressor.fit(x_train, y_train)

        adapter = _RegressionShapTreeAdapter()
        importance = adapter.compute_importance(
            model=regressor,
            x_data=x_train,
            feature_names=["feat_a", "feat_b"],
        )

        assert len(importance) == 2
        assert importance[0]["rank"] == 1
        assert importance[1]["rank"] == 2
        assert importance[0]["importance"] >= 0.0

    def test_compute_importance_with_wrapped_xgboost(self) -> None:
        """SHAP tree adapter unwraps _XGBRegressorPrepared to get raw model."""
        from covenant_ml.backends.xgboost.regressor import _XGBRegressorPrepared
        from covenant_ml.explainers.regression_adapters import _RegressionShapTreeAdapter
        from covenant_ml.types_regression import XGBRegressorModelProtocol

        xgb_mod = __import__("xgboost")
        xgb_model: XGBRegressorModelProtocol = xgb_mod.XGBRegressor(
            n_estimators=5,
            max_depth=2,
            random_state=42,
        )

        x_train = _make_x([[1.0, 2.0], [3.0, 1.0], [5.0, 0.5], [2.0, 4.0]])
        y_train: NDArray[np.float64] = np.arange(1.0, 5.0, dtype=np.float64)
        xgb_model.fit(x_train, y_train)

        # Wrap in _XGBRegressorPrepared (same as backend.load() returns)
        wrapped = _XGBRegressorPrepared(xgb_model)

        adapter = _RegressionShapTreeAdapter()
        importance = adapter.compute_importance(
            model=wrapped,
            x_data=x_train,
            feature_names=["feat_a", "feat_b"],
        )

        assert len(importance) == 2
        assert importance[0]["rank"] == 1
        assert importance[1]["rank"] == 2
        assert importance[0]["importance"] >= 0.0
