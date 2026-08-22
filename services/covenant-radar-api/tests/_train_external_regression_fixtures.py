"""Shared fixtures and helpers for test_train_external_regression_job splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from covenant_ml.backends.protocol import BackendCapabilities
from covenant_ml.backends.regressor_protocol import (
    PreparedRegressor,
    RegressorBackend,
    RegressorProgressCallback,
)
from covenant_ml.backends.regressor_registry import (
    RegressorBackendRegistration,
    RegressorRegistry,
)
from covenant_ml.datasets import (
    RegressionDatasetConfig,
    RegressionDatasetRegistry,
    RegressionLoadedDataset,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import RegressionDatasetMeta, RegressionTargetSpec
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SearchSpace,
)
from covenant_ml.types import (
    FeatureImportance,
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressorBackendName,
    RegressorTrainConfig,
)
from numpy.typing import NDArray


def _make_fake_regression_dataset(
    name: str = "financial_distress",
) -> RegressionLoadedDataset:
    """Create fake regression dataset for testing.

    Args:
        name: Dataset name.

    Returns:
        RegressionLoadedDataset with synthetic data.
    """
    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((80, 6)).astype(np.float64)
    y: NDArray[np.float64] = rng.random(80).astype(np.float64)
    meta: RegressionDatasetMeta = {
        "name": name,
        "n_samples": 80,
        "n_features": 6,
        "feature_names": tuple(f"feature_{i}" for i in range(6)),
        "target_mean": 0.5,
        "target_std": 0.3,
        "target_min": 0.0,
        "target_max": 1.0,
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y}


def _make_fake_regression_config(name: str) -> RegressionDatasetConfig:
    """Create fake regression dataset config.

    Args:
        name: Dataset name.

    Returns:
        RegressionDatasetConfig for testing.
    """
    return RegressionDatasetConfig(
        name=name,
        display_name=f"Fake {name}",
        folder=f"{name}_data",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=RegressionTargetSpec(column_name="target"),
        exclude_columns=(),
        n_samples_expected=80,
        n_features_expected=6,
        target_mean_expected=0.5,
    )


def _make_fake_regression_registry() -> RegressionDatasetRegistry:
    """Create fake regression dataset registry.

    Returns:
        RegressionDatasetRegistry with financial_distress.
    """
    configs = (_make_fake_regression_config("financial_distress"),)
    return RegressionDatasetRegistry(configs)


def _make_fake_regression_loader(
    config: RegressionDatasetConfig,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> RegressionLoadedDataset:
    """Fake regression dataset loader that returns synthetic data.

    Args:
        config: Dataset config (used for name).
        external_dir: Ignored.
        progress_callback: Ignored.

    Returns:
        Fake RegressionLoadedDataset.
    """
    return _make_fake_regression_dataset(config["name"])


def _make_fake_metrics() -> RegressionMetrics:
    """Create fake regression metrics.

    Returns:
        RegressionMetrics with fixed values.
    """
    return {
        "mse": 0.01,
        "rmse": 0.1,
        "mae": 0.08,
        "r_squared": 0.95,
        "mape": 5.0,
    }


def _make_fake_feature_importances() -> list[FeatureImportance]:
    """Create fake feature importances.

    Returns:
        List of 6 FeatureImportance dicts.
    """
    return [{"name": f"feature_{i}", "importance": 0.2 - i * 0.03, "rank": i + 1} for i in range(6)]


class _FakePreparedRegressor:
    """Fake prepared regressor for testing."""

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return zeros.

        Args:
            x: Feature matrix.

        Returns:
            Zero predictions.
        """
        n: int = int(x.shape[0])
        return np.zeros(n, dtype=np.float64)


class _FakeRegressorBackend:
    """Fake regressor backend for testing.

    Returns predetermined training outcomes without training.
    """

    def __init__(
        self,
        name: RegressorBackendName = "xgboost_reg",
    ) -> None:
        self._name = name
        self.train_calls: list[RegressorTrainConfig] = []

    def backend_name(self) -> RegressorBackendName:
        """Return the backend name.

        Returns:
            Backend name literal.
        """
        return self._name

    def capabilities(self) -> BackendCapabilities:
        """Return capabilities.

        Returns:
            Backend capabilities dict.
        """
        return {
            "supports_train": True,
            "supports_gpu": True,
            "supports_early_stopping": True,
            "supports_feature_importance": True,
            "model_format": "ubj",
        }

    def prepare(
        self,
        *,
        n_features: int,
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        """Not supported.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("Not supported")

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
        """Return a predetermined outcome.

        Args:
            x_features: Feature matrix (recorded but not used).
            y_targets: Target values (recorded but not used).
            feature_names: Feature names.
            config: Training config (recorded).
            output_dir: Directory to write fake model.
            progress: Ignored.

        Returns:
            Predetermined RegressionTrainOutcome.
        """
        self.train_calls.append(config)

        # Write a fake model file
        model_path = output_dir / "fake_model.ubj"
        model_path.write_bytes(b"fake-model-data")

        return {
            "model_path": str(model_path),
            "model_id": self._name,
            "samples_total": 80,
            "samples_train": 56,
            "samples_val": 12,
            "samples_test": 12,
            "train_metrics": _make_fake_metrics(),
            "val_metrics": _make_fake_metrics(),
            "test_metrics": _make_fake_metrics(),
            "best_val_rmse": 0.1,
            "best_round": 8,
            "total_rounds": 10,
            "early_stopped": True,
            "config": config,
            "feature_importances": _make_fake_feature_importances(),
        }

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        """Return fake metrics.

        Returns:
            Fake RegressionMetrics.
        """
        return _make_fake_metrics()

    def save(self, *, model: PreparedRegressor, path: str) -> None:
        """Not supported.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("Not supported")

    def load(self, *, path: str) -> PreparedRegressor:
        """Return fake prepared regressor.

        Returns:
            FakePreparedRegressor instance.
        """
        return _FakePreparedRegressor()

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Not supported.

        Returns:
            None.
        """
        return None

    def get_default_search_space(self) -> SearchSpace:
        """Not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Not supported.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


def _make_fake_regressor_registry(
    backend_name: RegressorBackendName = "xgboost_reg",
) -> tuple[RegressorRegistry, _FakeRegressorBackend]:
    """Create fake regressor registry with one backend.

    Args:
        backend_name: Backend name to register.

    Returns:
        Tuple of (registry, fake_backend).
    """
    fake_backend = _FakeRegressorBackend(backend_name)

    def _factory() -> RegressorBackend:
        return fake_backend

    reg = RegressorRegistry()
    reg.register(
        backend_name,
        RegressorBackendRegistration(_factory),
    )
    return reg, fake_backend
