"""Tests for worker/train_external_regression_job.py.

Tests use dependency injection via worker/_regression_hooks to verify actual
code paths. All code paths are tested with strong assertions on actual behavior.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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
from platform_core.json_utils import (
    dump_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)
from platform_core.testing import FakeEnv

from covenant_radar_api.worker import _regression_hooks as reg_hooks
from covenant_radar_api.worker import _test_hooks as worker_hooks
from covenant_radar_api.worker.train_external_regression_job import (
    _build_lightgbm_reg_log,
    _build_xgboost_reg_log,
    _dispatch_regression_backend,
    _get_regression_active_filename,
    _get_regression_meta_filename,
    _importance_to_json,
    _regression_metrics_to_json,
    _write_regression_model_metadata,
    process_external_regression_train_job,
    run_external_regression_training,
)

# =============================================================================
# Fake Implementations for Testing
# =============================================================================


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


# =============================================================================
# Tests: JSON serialization helpers
# =============================================================================


class TestRegressionMetricsToJson:
    """Tests for _regression_metrics_to_json."""

    def test_converts_all_fields(self) -> None:
        """All RegressionMetrics fields are converted."""
        metrics = _make_fake_metrics()
        result = _regression_metrics_to_json(metrics)

        assert result["mse"] == 0.01
        assert result["rmse"] == 0.1
        assert result["mae"] == 0.08
        assert result["r_squared"] == 0.95
        assert result["mape"] == 5.0

    def test_returns_five_keys(self) -> None:
        """Result has exactly 5 keys."""
        metrics = _make_fake_metrics()
        result = _regression_metrics_to_json(metrics)
        assert len(result) == 5


class TestImportanceToJson:
    """Tests for _importance_to_json."""

    def test_converts_importance(self) -> None:
        """Converts FeatureImportance to JSON dict."""
        imp: FeatureImportance = {
            "name": "feature_0",
            "importance": 0.25,
            "rank": 1,
        }
        result = _importance_to_json(imp)
        assert result["name"] == "feature_0"
        assert result["importance"] == 0.25
        assert result["rank"] == 1


# =============================================================================
# Tests: Config log builders
# =============================================================================


class TestBuildXGBoostRegLog:
    """Tests for _build_xgboost_reg_log."""

    def test_extracts_key_params(self) -> None:
        """Extracts key XGBoost parameters for logging."""
        from covenant_ml.types import TrainConfig

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
            "early_stopping_rounds": 10,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        result = _build_xgboost_reg_log(config)
        assert result["learning_rate"] == 0.1
        assert result["n_estimators"] == 10
        assert result["max_depth"] == 3
        assert len(result) == 5


class TestBuildLightGBMRegLog:
    """Tests for _build_lightgbm_reg_log."""

    def test_extracts_key_params(self) -> None:
        """Extracts key LightGBM parameters for logging."""
        from covenant_ml.types import LightGBMConfig

        config: LightGBMConfig = {
            "device": "cpu",
            "learning_rate": 0.05,
            "max_depth": 5,
            "n_estimators": 100,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": 42,
            "early_stopping_rounds": 10,
        }
        result = _build_lightgbm_reg_log(config)
        assert result["num_leaves"] == 31
        assert result["reg_alpha"] == 0.1
        assert len(result) == 6


# =============================================================================
# Tests: Active filename and metadata
# =============================================================================


class TestGetRegressionActiveFilename:
    """Tests for _get_regression_active_filename."""

    def test_xgboost_reg(self) -> None:
        """XGBoost regressor returns .ubj filename."""
        assert _get_regression_active_filename("xgboost_reg") == "active_xgb_reg.ubj"

    def test_lightgbm_reg(self) -> None:
        """LightGBM regressor returns .txt filename."""
        assert _get_regression_active_filename("lightgbm_reg") == "active_lgbm_reg.txt"

    def test_unknown_raises_value_error(self) -> None:
        """Unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown regressor backend"):
            _get_regression_active_filename("mlp_reg")


class TestGetRegressionMetaFilename:
    """Tests for _get_regression_meta_filename."""

    def test_xgboost_reg_empty(self) -> None:
        """XGBoost regressor has no metadata (self-describing)."""
        assert _get_regression_meta_filename("xgboost_reg") == ""

    def test_lightgbm_reg(self) -> None:
        """LightGBM regressor has metadata."""
        assert _get_regression_meta_filename("lightgbm_reg") == "active_lgbm_reg_meta.json"


class TestWriteRegressionModelMetadata:
    """Tests for _write_regression_model_metadata."""

    def test_xgboost_reg_returns_none(self, tmp_path: Path) -> None:
        """XGBoost regressor returns None (no metadata needed)."""
        result = _write_regression_model_metadata("xgboost_reg", tmp_path)
        assert result is None

    def test_lightgbm_reg_writes_metadata(self, tmp_path: Path) -> None:
        """LightGBM regressor writes metadata file."""
        result = _write_regression_model_metadata("lightgbm_reg", tmp_path)
        expected_path = tmp_path / "active_lgbm_reg_meta.json"
        assert result == expected_path
        assert expected_path.exists()

        content = expected_path.read_text(encoding="utf-8")
        assert '"backend": "lightgbm_reg"' in content


# =============================================================================
# Tests: Dispatch regression backend
# =============================================================================


class TestDispatchRegressionBackend:
    """Tests for _dispatch_regression_backend."""

    def test_xgboost_reg_dispatch(self) -> None:
        """XGBoost regressor dispatch returns log dict."""
        from covenant_ml.types import TrainConfig

        from covenant_radar_api.worker._train_external_regression_parsers import (
            XGBoostRegParseResult,
        )

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
            "early_stopping_rounds": 10,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        parse_result: XGBoostRegParseResult = {
            "backend": "xgboost_reg",
            "config": config,
            "dataset": "financial_distress",
        }
        log_dict = _dispatch_regression_backend(parse_result)
        assert log_dict["learning_rate"] == 0.1

    def test_lightgbm_reg_dispatch(self) -> None:
        """LightGBM regressor dispatch returns log dict."""
        from covenant_ml.types import LightGBMConfig

        from covenant_radar_api.worker._train_external_regression_parsers import (
            LightGBMRegParseResult,
        )

        config: LightGBMConfig = {
            "device": "cpu",
            "learning_rate": 0.05,
            "max_depth": 5,
            "n_estimators": 100,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": 42,
            "early_stopping_rounds": 10,
        }
        parse_result: LightGBMRegParseResult = {
            "backend": "lightgbm_reg",
            "config": config,
            "dataset": "financial_distress",
        }
        log_dict = _dispatch_regression_backend(parse_result)
        assert log_dict["num_leaves"] == 31


# =============================================================================
# Tests: run_external_regression_training
# =============================================================================


class TestRunExternalRegressionTraining:
    """Tests for run_external_regression_training."""

    def setup_method(self) -> None:
        """Install fake worker_hooks."""
        self._orig_dataset_registry = reg_hooks.regression_registry_factory
        self._orig_dataset_loader = reg_hooks.regression_dataset_loader
        self._orig_regressor_registry = reg_hooks.regressor_registry_factory

        reg_hooks.regression_registry_factory = _make_fake_regression_registry
        reg_hooks.regression_dataset_loader = _make_fake_regression_loader

        registry, self._fake_backend = _make_fake_regressor_registry("xgboost_reg")

        def _reg_factory() -> RegressorRegistry:
            return registry

        reg_hooks.regressor_registry_factory = _reg_factory

    def teardown_method(self) -> None:
        """Restore original worker_hooks."""
        reg_hooks.regression_registry_factory = self._orig_dataset_registry
        reg_hooks.regression_dataset_loader = self._orig_dataset_loader
        reg_hooks.regressor_registry_factory = self._orig_regressor_registry

    def test_xgboost_reg_produces_result(self, tmp_path: Path) -> None:
        """XGBoost regression training produces expected result."""
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        output_dir = tmp_path / "models"
        output_dir.mkdir()

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )

        result = run_external_regression_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["dataset"] == "financial_distress"
        assert result["backend"] == "xgboost_reg"
        assert result["n_features"] == 6
        assert result["samples_total"] == 80
        assert result["best_val_rmse"] == 0.1
        assert result["early_stopped"] is True

        # Verify active model file was created
        active_path = Path(str(result["active_model_path"]))
        assert active_path.exists()
        assert active_path.name == "active_xgb_reg.ubj"

        # No metadata for XGBoost (self-describing)
        assert result["active_meta_path"] is None

        # Verify metrics structure
        train_metrics = narrow_json_to_dict(result["train_metrics"])
        assert require_float(train_metrics, "rmse") == 0.1
        assert require_float(train_metrics, "r_squared") == 0.95

        # Verify feature importances
        importances = result["feature_importances"]
        assert type(importances) is list
        assert len(importances) == 6
        first_imp = narrow_json_to_dict(importances[0])
        assert require_int(first_imp, "rank") == 1
        assert require_str(first_imp, "name") == "feature_0"

    def test_lightgbm_reg_produces_result(self, tmp_path: Path) -> None:
        """LightGBM regression training produces metadata file."""
        # Set up lightgbm_reg backend
        lgbm_registry, _lgbm_backend = _make_fake_regressor_registry("lightgbm_reg")

        def _lgbm_factory() -> RegressorRegistry:
            return lgbm_registry

        reg_hooks.regressor_registry_factory = _lgbm_factory

        external_dir = tmp_path / "external"
        external_dir.mkdir()
        output_dir = tmp_path / "models"
        output_dir.mkdir()

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "backend": "lightgbm_reg",
                "device": "cpu",
                "learning_rate": 0.05,
                "max_depth": 5,
                "n_estimators": 100,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )

        result = run_external_regression_training(config_json, external_dir, output_dir)

        assert result["status"] == "complete"
        assert result["backend"] == "lightgbm_reg"

        # LightGBM has metadata
        meta_path_str = result["active_meta_path"]
        assert type(meta_path_str) is str
        meta_path = Path(meta_path_str)
        assert meta_path.exists()
        assert meta_path.name == "active_lgbm_reg_meta.json"

        # Verify active model file
        active_path = Path(str(result["active_model_path"]))
        assert active_path.exists()
        assert active_path.name == "active_lgbm_reg.txt"

    def test_backend_receives_config(self, tmp_path: Path) -> None:
        """Backend train() is called with parsed config."""
        external_dir = tmp_path / "external"
        external_dir.mkdir()
        output_dir = tmp_path / "models"
        output_dir.mkdir()

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "learning_rate": 0.2,
                "max_depth": 5,
                "n_estimators": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 99,
            }
        )

        run_external_regression_training(config_json, external_dir, output_dir)

        assert len(self._fake_backend.train_calls) == 1
        raw_config = self._fake_backend.train_calls[0]
        # All RegressorTrainConfig variants have learning_rate and train_ratio
        assert raw_config["learning_rate"] == 0.2
        assert raw_config["train_ratio"] == 0.7


# =============================================================================
# Tests: process_external_regression_train_job (RQ entry point)
# =============================================================================


class TestProcessExternalRegressionTrainJob:
    """Tests for process_external_regression_train_job."""

    def setup_method(self) -> None:
        """Install fake worker_hooks."""
        self._orig_dataset_registry = reg_hooks.regression_registry_factory
        self._orig_dataset_loader = reg_hooks.regression_dataset_loader
        self._orig_regressor_registry = reg_hooks.regressor_registry_factory
        self._orig_data_bank = worker_hooks.data_bank_uploader

        reg_hooks.regression_registry_factory = _make_fake_regression_registry
        reg_hooks.regression_dataset_loader = _make_fake_regression_loader

        registry, self._fake_backend = _make_fake_regressor_registry("xgboost_reg")

        def _reg_factory() -> RegressorRegistry:
            return registry

        reg_hooks.regressor_registry_factory = _reg_factory

    def teardown_method(self) -> None:
        """Restore original worker_hooks."""
        reg_hooks.regression_registry_factory = self._orig_dataset_registry
        reg_hooks.regression_dataset_loader = self._orig_dataset_loader
        reg_hooks.regressor_registry_factory = self._orig_regressor_registry
        worker_hooks.data_bank_uploader = self._orig_data_bank

    def test_process_job_without_data_bank(self, tmp_path: Path) -> None:
        """RQ entry point works without data-bank config."""
        data_root = tmp_path / "data"
        external_dir = data_root / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        models_dir = tmp_path / "models"

        fake_env = FakeEnv(
            {
                "APP__DATA_ROOT": str(data_root),
                "APP__MODELS_ROOT": str(models_dir),
                "DATABASE_URL": "postgresql://test@localhost/test",
                "REDIS_URL": "redis://localhost:6379/0",
            }
        )

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )

        from platform_core.config import _test_hooks as config_hooks

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env
        try:
            result = process_external_regression_train_job(config_json)
        finally:
            config_hooks.get_env = orig_get_env

        assert result["status"] == "complete"
        assert result["dataset"] == "financial_distress"
        assert result["backend"] == "xgboost_reg"

    def test_process_job_with_data_bank(self, tmp_path: Path) -> None:
        """RQ entry point uploads to data-bank when configured."""
        upload_calls: list[str] = []

        def _fake_uploader(
            model_path: Path,
            data_bank_url: str,
            data_bank_key: str,
        ) -> str:
            upload_calls.append(str(model_path))
            return "fake-file-id-123"

        worker_hooks.data_bank_uploader = _fake_uploader

        data_root = tmp_path / "data"
        external_dir = data_root / "external"
        external_dir.mkdir(parents=True, exist_ok=True)
        models_dir = tmp_path / "models"

        fake_env = FakeEnv(
            {
                "APP__DATA_ROOT": str(data_root),
                "APP__MODELS_ROOT": str(models_dir),
                "DATA_BANK_API_URL": "https://databank.example.com",
                "DATA_BANK_API_KEY": "test-api-key",
                "DATABASE_URL": "postgresql://test@localhost/test",
                "REDIS_URL": "redis://localhost:6379/0",
            }
        )

        config_json = dump_json_str(
            {
                "dataset": "financial_distress",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )

        from platform_core.config import _test_hooks as config_hooks

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env
        try:
            result = process_external_regression_train_job(config_json)
        finally:
            config_hooks.get_env = orig_get_env

        assert result["status"] == "complete"
        assert result["model_file_id"] == "fake-file-id-123"
        assert len(upload_calls) == 1
