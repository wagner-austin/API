"""Tests for process_external_train_job RQ entry point."""

from __future__ import annotations

from pathlib import Path

from platform_core.config import _test_hooks as config_hooks
from platform_core.json_utils import dump_json_str
from platform_core.testing import FakeEnv

from covenant_radar_api.worker.train_external_job import process_external_train_job

from .conftest import write_taiwan_dataset


class TestProcessExternalTrainJob:
    """Tests for process_external_train_job RQ entry point."""

    def test_loads_settings_and_runs(self, tmp_path: Path) -> None:
        """process_external_train_job loads settings from env and runs training."""
        data_root = tmp_path / "data"
        external_dir = data_root / "external"
        models_dir = tmp_path / "models"

        write_taiwan_dataset(external_dir)

        fake_env = FakeEnv(
            {
                "APP__DATA_ROOT": str(data_root),
                "APP__MODELS_ROOT": str(models_dir),
                "DATABASE_URL": "postgresql://test@localhost/test",
                "REDIS_URL": "redis://localhost:6379/0",
            }
        )

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env

        try:
            config_json = dump_json_str(
                {
                    "dataset": "taiwan",
                    "learning_rate": 0.3,
                    "max_depth": 3,
                    "n_estimators": 10,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "random_state": 42,
                }
            )

            result = process_external_train_job(config_json)

            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"

            model_path = Path(str(result["model_path"]))
            assert model_path.exists()
        finally:
            config_hooks.get_env = orig_get_env
