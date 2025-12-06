from __future__ import annotations

from typing import Literal, Protocol

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.compute import LocalCPUProvider


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


def test_local_cpu_provider_uses_configured_threads(
    settings_factory: _SettingsFactory,
) -> None:
    settings: Settings = settings_factory(threads=2)
    threads: int = settings["app"]["threads"] if settings["app"]["threads"] > 0 else 1
    env = LocalCPUProvider(threads_count=threads).env()
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"
