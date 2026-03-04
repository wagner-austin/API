"""Tests for regressor backend registry.

Covers registry construction, registration, lookup, and capabilities
caching. Uses a fake backend — no real ML frameworks needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.backends.protocol import BackendCapabilities
from covenant_ml.backends.regressor_protocol import (
    PreparedRegressor,
    RegressorProgressCallback,
)
from covenant_ml.backends.regressor_registry import (
    RegressorBackendRegistration,
    RegressorRegistry,
    default_regressor_registry,
)
from covenant_ml.types import (
    FeatureImportance,
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressorBackendName,
    RegressorTrainConfig,
)


class _FakePreparedRegressor:
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros(x.shape[0], dtype=np.float64)


def _make_zero_metrics() -> RegressionMetrics:
    return RegressionMetrics(mse=0.0, rmse=0.0, mae=0.0, r_squared=0.0, mape=0.0)


class _FakeRegressorBackend:
    def backend_name(self) -> RegressorBackendName:
        return "xgboost_reg"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_train=True,
            supports_gpu=False,
            supports_early_stopping=True,
            supports_feature_importance=True,
            model_format="ubj",
        )

    def prepare(
        self,
        *,
        n_features: int,
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        _ = n_features, feature_names
        return _FakePreparedRegressor()

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str] | None,
        config: RegressorTrainConfig,
        output_dir: Path,
        progress: RegressorProgressCallback | None,
    ) -> RegressionTrainOutcome:
        _ = x_features, y_targets, feature_names, output_dir, progress
        m = _make_zero_metrics()
        return RegressionTrainOutcome(
            model_path="",
            model_id="",
            samples_total=0,
            samples_train=0,
            samples_val=0,
            samples_test=0,
            train_metrics=m,
            val_metrics=m,
            test_metrics=m,
            best_val_rmse=0.0,
            best_round=0,
            total_rounds=0,
            early_stopped=False,
            config=config,
            feature_importances=[],
        )

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        _ = model, x, y
        return _make_zero_metrics()

    def save(self, *, model: PreparedRegressor, path: str) -> None:
        _ = model, path

    def load(self, *, path: str) -> PreparedRegressor:
        _ = path
        return _FakePreparedRegressor()

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        _ = model, feature_names
        return None


def _create_fake_backend() -> _FakeRegressorBackend:
    return _FakeRegressorBackend()


def test_default_regressor_registry_has_xgboost_reg() -> None:
    """Default regressor registry includes xgboost_reg."""
    reg = default_regressor_registry()
    names = reg.list_backends()
    assert "xgboost_reg" in names


def test_default_regressor_registry_has_lightgbm_reg() -> None:
    """Default regressor registry includes lightgbm_reg."""
    reg = default_regressor_registry()
    names = reg.list_backends()
    assert "lightgbm_reg" in names


def test_registry_register_and_list() -> None:
    """Registered backends appear in list_backends."""
    reg = RegressorRegistry()
    reg.register("xgboost_reg", RegressorBackendRegistration(_create_fake_backend))

    names = reg.list_backends()
    assert names == ["xgboost_reg"]


def test_registry_register_multiple() -> None:
    """Multiple backends register and list in sorted order."""
    reg = RegressorRegistry()
    reg.register("xgboost_reg", RegressorBackendRegistration(_create_fake_backend))
    reg.register("mlp_reg", RegressorBackendRegistration(_create_fake_backend))

    names = reg.list_backends()
    assert names == ["mlp_reg", "xgboost_reg"]


def test_registry_get_returns_backend() -> None:
    """get() returns a working backend instance."""
    reg = RegressorRegistry()
    reg.register("xgboost_reg", RegressorBackendRegistration(_create_fake_backend))

    backend = reg.get("xgboost_reg")
    assert backend.backend_name() == "xgboost_reg"


def test_registry_get_capabilities() -> None:
    """get_capabilities returns BackendCapabilities."""
    reg = RegressorRegistry()
    reg.register("xgboost_reg", RegressorBackendRegistration(_create_fake_backend))

    caps = reg.get_capabilities("xgboost_reg")
    assert caps["supports_train"] is True
    assert caps["model_format"] == "ubj"


def test_registration_factory_returns_callable() -> None:
    """RegressorBackendRegistration.factory() returns the factory."""
    registration = RegressorBackendRegistration(_create_fake_backend)
    factory = registration.factory()
    assert callable(factory)
    backend = factory()
    assert backend.backend_name() == "xgboost_reg"


def test_registration_capabilities_caching() -> None:
    """Capabilities are cached after first access."""
    call_count = 0

    def counting_factory() -> _FakeRegressorBackend:
        nonlocal call_count
        call_count += 1
        return _FakeRegressorBackend()

    registration = RegressorBackendRegistration(counting_factory)

    # First access creates a backend to query capabilities
    caps1 = registration.capabilities()
    assert call_count == 1

    # Second access uses cache — no new backend created
    caps2 = registration.capabilities()
    assert call_count == 1
    assert caps1 == caps2


def test_registry_get_raises_on_missing_backend() -> None:
    """get() raises KeyError for unregistered backend."""
    import pytest

    reg = RegressorRegistry()
    with pytest.raises(KeyError):
        reg.get("xgboost_reg")


def test_registry_get_capabilities_raises_on_missing() -> None:
    """get_capabilities() raises KeyError for unregistered backend."""
    import pytest

    reg = RegressorRegistry()
    with pytest.raises(KeyError):
        reg.get_capabilities("mlp_reg")
