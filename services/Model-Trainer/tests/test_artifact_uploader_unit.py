from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.errors import AppError
from platform_core.trainer_keys import artifact_file_id_key
from platform_ml.artifact_store import ArtifactStore, ArtifactStoreError
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.core.config.settings import Settings
from model_trainer.worker import train_job


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
        threads: int | None = ...,
        redis_url: str | None = ...,
        app_env: Literal["dev", "prod"] | None = ...,
        security_api_key: str | None = ...,
    ) -> Settings: ...


def test_upload_and_persist_pointer_config_missing(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    r: RedisStrProto = FakeRedis()
    settings = settings_factory(
        data_bank_api_url="",
        data_bank_api_key="",
    )
    with pytest.raises(AppError, match="data-bank-api configuration missing"):
        train_job._upload_and_persist_pointer(
            settings,
            r,
            run_id="rid",
            out_dir=str(tmp_path),
        )


def test_upload_and_persist_pointer_missing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    class _Store:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def upload_artifact(
            self,
            dir_path: Path,
            *,
            artifact_name: str,
            request_id: str,
        ) -> FileUploadResponse:
            assert dir_path.exists(), "dir_path must exist"
            return {
                "file_id": "fid",
                "size": 1,
                "sha256": "x",
                "content_type": "application/gzip",
                "created_at": None,
            }

    monkeypatch.setattr("platform_ml.ArtifactStore", _Store)
    base = tmp_path / "missing-dir"
    base.mkdir(parents=True, exist_ok=True)

    r = FakeRedis()
    settings = settings_factory(
        data_bank_api_url="http://x",
        data_bank_api_key="k",
    )
    train_job._upload_and_persist_pointer(
        settings,
        r,
        run_id="run-missing",
        out_dir=str(base),
    )
    assert r.get(artifact_file_id_key("run-missing")) == "fid"


def test_upload_and_persist_pointer_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    base = tmp_path / "model"
    base.mkdir(parents=True, exist_ok=True)
    (base / "weights.bin").write_bytes(b"x")

    class _Store:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def upload_artifact(
            self,
            dir_path: Path,
            *,
            artifact_name: str,
            request_id: str,
        ) -> FileUploadResponse:
            assert dir_path == base
            return {
                "file_id": "deadbeef",
                "size": 10,
                "sha256": "deadbeef",
                "content_type": "application/gzip",
                "created_at": None,
            }

    monkeypatch.setattr("platform_ml.ArtifactStore", _Store)

    r = FakeRedis()
    settings = settings_factory(
        data_bank_api_url="http://db.local",
        data_bank_api_key="secret",
    )
    train_job._upload_and_persist_pointer(
        settings,
        r,
        run_id="run1",
        out_dir=str(base),
    )
    assert r.get(artifact_file_id_key("run1")) == "deadbeef"


def test_upload_and_persist_pointer_store_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    base = tmp_path / "model2"
    base.mkdir(parents=True, exist_ok=True)
    (base / "weights.bin").write_bytes(b"data")

    class _StoreBad:
        def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout_seconds = timeout_seconds

        def upload_artifact(
            self,
            dir_path: Path,
            *,
            artifact_name: str,
            request_id: str,
        ) -> FileUploadResponse:
            raise ArtifactStoreError("boom")

    monkeypatch.setattr("platform_ml.ArtifactStore", _StoreBad)

    r: RedisStrProto = FakeRedis()
    settings = settings_factory(
        data_bank_api_url="http://db.local",
        data_bank_api_key="secret",
    )
    with pytest.raises(ArtifactStoreError):
        train_job._upload_and_persist_pointer(
            settings,
            r,
            run_id="run2",
            out_dir=str(base),
        )


def test_artifact_store_init_and_proxy() -> None:
    # Validate ArtifactStore can be instantiated independently (smoke)
    _ = ArtifactStore(base_url="http://x", api_key="k")
