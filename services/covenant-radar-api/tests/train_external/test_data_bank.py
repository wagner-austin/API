"""Tests for data-bank integration in training jobs."""

from __future__ import annotations

from pathlib import Path

from platform_core.config import _test_hooks as config_hooks
from platform_core.json_utils import dump_json_str
from platform_core.testing import FakeEnv

from covenant_radar_api.worker import _test_hooks as worker_hooks
from covenant_radar_api.worker.train_external_job import (
    _upload_model_to_data_bank,
    process_external_train_job,
)

from .conftest import write_taiwan_dataset


class TestUploadModelToDataBank:
    """Tests for _upload_model_to_data_bank function."""

    def test_success(self, tmp_path: Path) -> None:
        """Test successful model upload to data-bank via hook."""
        model_path = tmp_path / "test_model.ubj"
        model_path.write_bytes(b"fake model bytes")

        upload_calls: list[tuple[Path, str, str]] = []

        class FakeUploader:
            """Fake uploader implementing DataBankUploaderProtocol."""

            def __call__(
                self,
                model_path: Path,
                data_bank_url: str,
                data_bank_key: str,
            ) -> str:
                upload_calls.append((model_path, data_bank_url, data_bank_key))
                return model_path.name

        worker_hooks.data_bank_uploader = FakeUploader()

        result = _upload_model_to_data_bank(
            model_path,
            "https://data-bank.example.com",
            "test-api-key",
        )

        assert result == "test_model.ubj"
        assert len(upload_calls) == 1
        assert upload_calls[0][0] == model_path
        assert upload_calls[0][1] == "https://data-bank.example.com"
        assert upload_calls[0][2] == "test-api-key"


class TestProcessJobWithDataBank:
    """Tests for process_external_train_job with data-bank integration."""

    def test_uploads_model_when_configured(self, tmp_path: Path) -> None:
        """Test that process job uploads model when data-bank is configured."""
        upload_calls: list[str] = []

        class FakeUploader:
            """Fake uploader that tracks calls."""

            def __call__(
                self,
                model_path: Path,
                data_bank_url: str,
                data_bank_key: str,
            ) -> str:
                upload_calls.append(model_path.name)
                return model_path.name

        worker_hooks.data_bank_uploader = FakeUploader()

        fake_env = FakeEnv(
            {
                "APP_ENV": "dev",
                "APP__DATA_ROOT": str(tmp_path / "data"),
                "APP__MODELS_ROOT": str(tmp_path / "models"),
                "DATA_BANK_API_URL": "https://data-bank.example.com",
                "DATA_BANK_API_KEY": "test-key",
                "DATABASE_URL": "postgresql://test:test@localhost/test",
                "REDIS_URL": "redis://localhost:6379",
            }
        )

        write_taiwan_dataset(tmp_path / "data" / "external")

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )

        result = process_external_train_job(config_json)

        config_hooks.get_env = orig_get_env

        assert result["status"] == "complete"
        assert "model_file_id" in result
        assert result["model_file_id"] == "active_xgb.ubj"
        assert len(upload_calls) == 1
        assert upload_calls[0] == "active_xgb.ubj"

    def test_no_upload_when_not_configured(self, tmp_path: Path) -> None:
        """Test that process job does not upload when data-bank is not configured."""
        upload_calls: list[str] = []

        class FakeUploader:
            """Fake uploader that tracks calls."""

            def __call__(
                self,
                model_path: Path,
                data_bank_url: str,
                data_bank_key: str,
            ) -> str:
                upload_calls.append(model_path.name)
                return model_path.name

        worker_hooks.data_bank_uploader = FakeUploader()

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        fake_env = FakeEnv(
            {
                "APP_ENV": "dev",
                "APP__DATA_ROOT": str(tmp_path / "data"),
                "APP__MODELS_ROOT": str(models_dir),
                "DATA_BANK_API_URL": "",
                "DATA_BANK_API_KEY": "",
                "DATABASE_URL": "postgresql://test:test@localhost/test",
                "REDIS_URL": "redis://localhost:6379",
            }
        )

        write_taiwan_dataset(tmp_path / "data" / "external")

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env

        config_json = dump_json_str(
            {
                "dataset": "taiwan",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
        )

        result = process_external_train_job(config_json)

        config_hooks.get_env = orig_get_env

        assert result["status"] == "complete"
        assert "model_file_id" not in result or result.get("model_file_id") is None
        assert len(upload_calls) == 0
        active_model_path = Path(str(result["active_model_path"]))
        assert active_model_path.exists()
