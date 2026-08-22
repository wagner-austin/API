"""Tests for worker/train_external_regression_job.py.

Tests use dependency injection via worker/_regression_hooks to verify actual
code paths. All code paths are tested with strong assertions on actual behavior.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.backends.regressor_registry import (
    RegressorRegistry,
)
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
    process_external_regression_train_job,
    run_external_regression_training,
)
from tests._train_external_regression_fixtures import (
    _make_fake_regression_loader,
    _make_fake_regression_registry,
    _make_fake_regressor_registry,
)


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
