from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_workers.testing import FakeRedis as FakeRedisClient

from model_trainer.core.config.settings import Settings, load_settings


class SettingsFactory(Protocol):
    def __call__(
        self: SettingsFactory,
        *,
        artifacts_root: str | None = None,
        runs_root: str | None = None,
        logs_root: str | None = None,
        data_root: str | None = None,
        data_bank_api_url: str | None = None,
        data_bank_api_key: str | None = None,
        threads: int | None = None,
        redis_url: str | None = None,
        app_env: Literal["dev", "prod"] | None = None,
        security_api_key: str | None = None,
    ) -> Settings: ...


def _make_fake_redis_client() -> FakeRedisClient:
    return FakeRedisClient()


fake_redis_client = pytest.fixture(_make_fake_redis_client)


def _build_settings(
    *,
    artifacts_root: str | None = None,
    runs_root: str | None = None,
    logs_root: str | None = None,
    data_root: str | None = None,
    data_bank_api_url: str | None = None,
    data_bank_api_key: str | None = None,
    threads: int | None = None,
    redis_url: str | None = None,
    app_env: Literal["dev", "prod"] | None = None,
    security_api_key: str | None = None,
) -> Settings:
    base = load_settings()
    _apply_app_overrides(
        base,
        artifacts_root=artifacts_root,
        runs_root=runs_root,
        logs_root=logs_root,
        data_root=data_root,
        data_bank_api_url=data_bank_api_url,
        data_bank_api_key=data_bank_api_key,
        threads=threads,
    )
    if redis_url is not None:
        base["redis"]["url"] = redis_url
    if app_env is not None:
        base["app_env"] = app_env
    if security_api_key is not None:
        base["security"]["api_key"] = security_api_key
    return base


def _apply_app_overrides(
    base: Settings,
    *,
    artifacts_root: str | None,
    runs_root: str | None,
    logs_root: str | None,
    data_root: str | None,
    data_bank_api_url: str | None,
    data_bank_api_key: str | None,
    threads: int | None,
) -> None:
    if artifacts_root is not None:
        base["app"]["artifacts_root"] = artifacts_root
    if runs_root is not None:
        base["app"]["runs_root"] = runs_root
    if logs_root is not None:
        base["app"]["logs_root"] = logs_root
    if data_root is not None:
        base["app"]["data_root"] = data_root
    if data_bank_api_url is not None:
        base["app"]["data_bank_api_url"] = data_bank_api_url
    if data_bank_api_key is not None:
        base["app"]["data_bank_api_key"] = data_bank_api_key
    if threads is not None:
        base["app"]["threads"] = threads


def _make_settings_factory() -> SettingsFactory:
    return _build_settings


settings_factory = pytest.fixture(_make_settings_factory)


def _make_settings_with_paths(tmp_path: Path, settings_factory: SettingsFactory) -> Settings:
    return settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )


settings_with_paths = pytest.fixture(_make_settings_with_paths)


@pytest.fixture(autouse=True)
def _patch_worker_settings(
    tmp_path: Path, settings_factory: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        redis_url="redis://localhost:6379/0",
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="test-key",
    )
    monkeypatch.setattr("model_trainer.worker.train_job.load_settings", lambda: settings)
    monkeypatch.setattr("model_trainer.worker.eval_job.load_settings", lambda: settings)
    monkeypatch.setattr("model_trainer.worker.score_job.load_settings", lambda: settings)
    monkeypatch.setattr("model_trainer.worker.generate_job.load_settings", lambda: settings)
    monkeypatch.setattr("model_trainer.worker.tokenizer_worker.load_settings", lambda: settings)
