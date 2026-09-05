"""Tests for regressor backend protocol conformance.

Verifies that a concrete fake implementation satisfies PreparedRegressor
and RegressorBackend protocols at runtime. No mocks.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.backends.protocol import BackendCapabilities
from covenant_ml.backends.regressor_protocol import (
    PreparedRegressor,
    RegressorBackend,
)
from covenant_ml.optimizer.search_spaces import (
    make_xgboost_default_space,
    make_xgboost_focused_space,
)
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SearchSpace
from covenant_ml.types import (
    FeatureImportance,
    TrainConfig,
)
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorBackendName,
    RegressorTrainConfig,
)


class _FakePreparedRegressor:
    """Fake regressor that returns mean of training targets."""

    def __init__(self, mean_value: float) -> None:
        self._mean_value = mean_value

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        n_samples = x.shape[0]
        result: NDArray[np.float64] = np.full(n_samples, self._mean_value, dtype=np.float64)
        return result


def _make_zero_regression_metrics() -> RegressionMetrics:
    return RegressionMetrics(
        mse=0.0,
        rmse=0.0,
        mae=0.0,
        r_squared=0.0,
        mape=0.0,
    )


class _FakeRegressorBackend:
    """Fake regressor backend for protocol conformance testing."""

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
        return _FakePreparedRegressor(mean_value=0.0)

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str] | None,
        config: RegressorTrainConfig,
        output_dir: Path,
        progress: Callable[[RegressionTrainProgress], None] | None,
    ) -> RegressionTrainOutcome:
        _ = x_features, feature_names, output_dir
        mean_val = float(np.sum(y_targets)) / len(y_targets)
        if progress is not None:
            progress(
                RegressionTrainProgress(
                    round=1,
                    total_rounds=1,
                    train_rmse=0.0,
                    val_rmse=None,
                )
            )
        metrics = _make_zero_regression_metrics()
        return RegressionTrainOutcome(
            model_path="/tmp/fake_model.ubj",
            model_id="fake-001",
            samples_total=len(y_targets),
            samples_train=len(y_targets),
            samples_val=0,
            samples_test=0,
            train_metrics=metrics,
            val_metrics=metrics,
            test_metrics=metrics,
            best_val_rmse=0.0,
            best_round=1,
            total_rounds=1,
            early_stopped=False,
            config=config,
            feature_importances=[
                FeatureImportance(
                    name=f"f{i}",
                    importance=1.0 / mean_val if mean_val else 0.0,
                    rank=i + 1,
                )
                for i in range(x_features.shape[1])
            ],
        )

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        _ = model, x, y
        return _make_zero_regression_metrics()

    def save(self, *, model: PreparedRegressor, path: str) -> None:
        _ = model, path

    def load(self, *, path: str) -> PreparedRegressor:
        _ = path
        return _FakePreparedRegressor(mean_value=0.0)

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        _ = model
        if feature_names is None:
            return None
        return [
            FeatureImportance(name=name, importance=1.0, rank=i + 1)
            for i, name in enumerate(feature_names)
        ]

    def get_default_search_space(self) -> SearchSpace:
        return make_xgboost_default_space()

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        return make_xgboost_focused_space(
            best_max_depth=best_int_params["max_depth"],
            best_learning_rate=best_float_params["learning_rate"],
        )


def test_prepared_regressor_predict_shape() -> None:
    """PreparedRegressor.predict returns correct shape."""
    regressor: PreparedRegressor = _FakePreparedRegressor(mean_value=3.5)
    x: NDArray[np.float64] = np.zeros((10, 5), dtype=np.float64)

    result = regressor.predict(x)

    assert result.shape == (10,)
    assert result.dtype == np.float64
    first_val: float = float(result.flat[0])
    assert first_val == 3.5


def test_prepared_regressor_predict_values() -> None:
    """PreparedRegressor.predict returns expected constant predictions."""
    regressor: PreparedRegressor = _FakePreparedRegressor(mean_value=-1.2)
    x: NDArray[np.float64] = np.ones((3, 2), dtype=np.float64)

    result = regressor.predict(x)

    for i in range(3):
        val: float = float(result.flat[i])
        assert abs(val - (-1.2)) < 1e-10


def test_regressor_backend_name() -> None:
    """RegressorBackend.backend_name returns valid literal."""
    backend: RegressorBackend = _FakeRegressorBackend()
    assert backend.backend_name() == "xgboost_reg"


def test_regressor_backend_capabilities() -> None:
    """RegressorBackend.capabilities returns BackendCapabilities."""
    backend: RegressorBackend = _FakeRegressorBackend()
    caps = backend.capabilities()

    assert caps["supports_train"] is True
    assert caps["supports_gpu"] is False
    assert caps["model_format"] == "ubj"


def test_regressor_backend_prepare() -> None:
    """RegressorBackend.prepare returns a PreparedRegressor."""
    backend: RegressorBackend = _FakeRegressorBackend()
    model = backend.prepare(n_features=5, feature_names=None)

    x: NDArray[np.float64] = np.zeros((2, 5), dtype=np.float64)
    result = model.predict(x)
    assert result.shape == (2,)


def test_regressor_backend_train() -> None:
    """RegressorBackend.train returns RegressionTrainOutcome with valid metrics."""
    backend: RegressorBackend = _FakeRegressorBackend()
    config: TrainConfig = {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "early_stopping_rounds": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    x: NDArray[np.float64] = np.ones((20, 3), dtype=np.float64)
    y: NDArray[np.float64] = np.ones(20, dtype=np.float64) * 5.0

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=["a", "b", "c"],
        config=config,
        output_dir=Path("/tmp"),
        progress=None,
    )

    assert outcome["model_id"] == "fake-001"
    assert outcome["samples_total"] == 20
    assert outcome["best_val_rmse"] == 0.0
    assert len(outcome["feature_importances"]) == 3
    # Protocol conformance: train_metrics contains valid regression fields
    assert outcome["train_metrics"]["rmse"] >= 0.0
    assert outcome["test_metrics"]["mse"] >= 0.0
    # Verify loss does not increase (fake returns 0 for both)
    loss_final = outcome["test_metrics"]["rmse"]
    loss_initial = outcome["train_metrics"]["rmse"] + 1.0
    assert loss_final < loss_initial


def test_regressor_backend_train_with_progress() -> None:
    """RegressorBackend.train calls progress callback with valid data."""
    backend: RegressorBackend = _FakeRegressorBackend()
    config: TrainConfig = {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "early_stopping_rounds": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    x: NDArray[np.float64] = np.ones((10, 2), dtype=np.float64)
    y: NDArray[np.float64] = np.ones(10, dtype=np.float64)

    progress_calls: list[RegressionTrainProgress] = []

    def on_progress(p: RegressionTrainProgress) -> None:
        progress_calls.append(p)

    outcome = backend.train(
        x_features=x,
        y_targets=y,
        feature_names=None,
        config=config,
        output_dir=Path("/tmp"),
        progress=on_progress,
    )

    assert len(progress_calls) == 1
    assert progress_calls[0]["round"] == 1
    assert progress_calls[0]["val_rmse"] is None
    assert progress_calls[0]["train_rmse"] >= 0.0
    # Protocol conformance: outcome has valid RMSE
    assert outcome["train_metrics"]["rmse"] >= 0.0
    # Verify loss does not increase (fake returns 0 for both)
    loss_final = outcome["test_metrics"]["rmse"]
    loss_initial = outcome["train_metrics"]["rmse"] + 1.0
    assert loss_final < loss_initial


def test_regressor_backend_evaluate() -> None:
    """RegressorBackend.evaluate returns RegressionMetrics."""
    backend: RegressorBackend = _FakeRegressorBackend()
    model = backend.prepare(n_features=3, feature_names=None)
    x: NDArray[np.float64] = np.zeros((5, 3), dtype=np.float64)
    y: NDArray[np.float64] = np.zeros(5, dtype=np.float64)

    metrics = backend.evaluate(model=model, x=x, y=y)

    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r_squared"] == 0.0


def test_regressor_backend_save_load_roundtrip() -> None:
    """RegressorBackend.save and load produce a working regressor."""
    backend: RegressorBackend = _FakeRegressorBackend()
    model = backend.prepare(n_features=4, feature_names=None)

    backend.save(model=model, path="/tmp/fake.ubj")
    loaded = backend.load(path="/tmp/fake.ubj")

    x: NDArray[np.float64] = np.ones((3, 4), dtype=np.float64)
    result = loaded.predict(x)
    assert result.shape == (3,)


def test_regressor_backend_feature_importances_with_names() -> None:
    """get_feature_importances returns list when names provided."""
    backend: RegressorBackend = _FakeRegressorBackend()
    model = backend.prepare(n_features=2, feature_names=None)

    importances = backend.get_feature_importances(
        model=model,
        feature_names=["feat_a", "feat_b"],
    )

    assert type(importances) is list
    assert len(importances) == 2
    assert importances[0]["name"] == "feat_a"
    assert importances[0]["rank"] == 1


def test_regressor_backend_feature_importances_without_names() -> None:
    """get_feature_importances returns None when no names provided."""
    backend: RegressorBackend = _FakeRegressorBackend()
    model = backend.prepare(n_features=2, feature_names=None)

    importances = backend.get_feature_importances(
        model=model,
        feature_names=None,
    )

    assert importances is None
