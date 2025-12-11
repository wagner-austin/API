from __future__ import annotations

from pathlib import Path
from typing import Literal, NoReturn, Protocol

import pytest
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TrainJobPayload
from model_trainer.worker import train_job
from model_trainer.worker.trainer_job_store import TrainerJobStore


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


class _FakeCorpusFetcher:
    """Fake CorpusFetcher for tests."""

    def __init__(self: _FakeCorpusFetcher, path: Path) -> None:
        self._path = path

    def fetch(self: _FakeCorpusFetcher, fid: str) -> Path:
        return self._path


def test_training_worker_failed_event_publish_branch(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Test that training errors set status to failed and propagate the exception."""
    # Fake redis via hook
    fake = FakeRedis()

    def _fake_kv_store(url: str) -> RedisStrProto:
        return fake

    _test_hooks.kv_store_factory = _fake_kv_store

    # Settings via hook
    settings = settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
    )

    def _load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _load_settings

    # Force container creation to raise so we enter exception handler
    def _fail_container(settings: Settings) -> NoReturn:
        raise RuntimeError("boom during container creation")

    _test_hooks.service_container_from_settings = _fail_container

    # Stub corpus fetcher via hook
    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return _FakeCorpusFetcher(tmp_path)

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    payload: TrainJobPayload = {
        "run_id": "run-err",
        "user_id": 1,
        "request": {
            "model_family": "gpt2",
            "model_size": "small",
            "max_seq_len": 16,
            "num_epochs": 1,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "tokenizer_id": "tok",
            "corpus_file_id": "deadbeef",
            "holdout_fraction": 0.01,
            "seed": 42,
            "pretrained_run_id": None,
            "freeze_embed": False,
            "gradient_clipping": 1.0,
            "optimizer": "adamw",
            "device": "cpu",
            "data_num_workers": None,
            "data_pin_memory": None,
            "early_stopping_patience": 5,
            "test_split_ratio": 0.15,
            "finetune_lr_cap": 5e-5,
            "precision": "auto",
        },
    }

    with pytest.raises(RuntimeError, match="boom during container creation"):
        train_job.process_train_job(payload)
    # Status is failed and message is set
    status = TrainerJobStore(fake).load("run-err")
    assert status is not None and status["status"] == "failed"
    assert status["message"] is not None and "boom during container creation" in status["message"]
    fake.assert_only_called({"set", "hset", "hgetall", "publish"})
