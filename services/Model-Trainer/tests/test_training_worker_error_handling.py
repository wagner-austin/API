"""Tests for training_worker.py error handling - covers lines 559 and 678."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NoReturn, Protocol

import pytest
from platform_workers.redis import RedisStrProto
from platform_workers.testing import (
    FakeRedisConditionalHsetError,
    FakeRedisConditionalHsetRedisError,
)

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.queue import TrainJobPayload
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


class _FakeCorpusFetcher:
    """Fake CorpusFetcher for tests."""

    def __init__(self: _FakeCorpusFetcher, corpus_root: Path, expected_fid: str) -> None:
        self._corpus_root = corpus_root
        self._expected_fid = expected_fid

    def fetch(self: _FakeCorpusFetcher, fid: str) -> Path:
        assert fid == self._expected_fid
        return self._corpus_root


def test_process_train_job_reraises_non_redis_error_on_handle_error(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Cover training_worker.py line 421 (non-Redis error re-raises during error handling).

    When training fails and then recording the error also fails with a non-Redis error,
    the non-Redis error should be re-raised (line 421).
    """
    artifacts_root = tmp_path / "artifacts"
    runs_root = tmp_path / "runs"
    logs_root = tmp_path / "logs"
    settings = settings_factory(
        artifacts_root=str(artifacts_root),
        runs_root=str(runs_root),
        logs_root=str(logs_root),
        data_root=str(tmp_path / "data"),
    )

    def _test_load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _test_load_settings

    payload: TrainJobPayload = {
        "run_id": "run-non-redis-err",
        "user_id": 7,
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
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "a.txt").write_text("hello\n", encoding="utf-8")

    # Create fake corpus fetcher via hook
    fake_fetcher = _FakeCorpusFetcher(corpus_root, "deadbeef")

    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return fake_fetcher

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    # Use specialized FakeRedis that fails with RuntimeError when status == "failed"
    client = FakeRedisConditionalHsetError(fail_on_status="failed")

    def _fake_kv_store(url: str) -> RedisStrProto:
        return client

    _test_hooks.kv_store_factory = _fake_kv_store

    # Force training to fail so we enter the error handling path
    def _fail_container(settings: Settings) -> NoReturn:
        raise RuntimeError("training failed")

    _test_hooks.service_container_from_settings = _fail_container

    # The non-Redis error during error handling should be re-raised (line 421)
    with pytest.raises(RuntimeError, match="simulated hset failure"):
        train_job.process_train_job(payload)

    client.assert_only_called({"hset", "publish", "set"})


def test_process_train_job_logs_redis_error_on_handle_error(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    """Cover training_worker.py line 678 (Redis error logs warning during error handling).

    When training fails and then recording the error also fails with a Redis error,
    the Redis error should be logged and the original exception re-raised.
    """
    artifacts_root = tmp_path / "artifacts"
    runs_root = tmp_path / "runs"
    logs_root = tmp_path / "logs"
    settings = settings_factory(
        artifacts_root=str(artifacts_root),
        runs_root=str(runs_root),
        logs_root=str(logs_root),
        data_root=str(tmp_path / "data"),
    )

    def _test_load_settings() -> Settings:
        return settings

    _test_hooks.load_settings = _test_load_settings

    payload: TrainJobPayload = {
        "run_id": "run-redis-err",
        "user_id": 8,
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
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "a.txt").write_text("hello\n", encoding="utf-8")

    # Create fake corpus fetcher via hook
    fake_fetcher = _FakeCorpusFetcher(corpus_root, "deadbeef")

    def _fake_corpus_fetcher_factory(
        api_url: str, api_key: str, cache_dir: Path
    ) -> _FakeCorpusFetcher:
        return fake_fetcher

    _test_hooks.corpus_fetcher_factory = _fake_corpus_fetcher_factory

    # Use specialized FakeRedis that fails with Redis error when status == "failed"
    client = FakeRedisConditionalHsetRedisError(fail_on_status="failed")

    def _fake_kv_store(url: str) -> RedisStrProto:
        return client

    _test_hooks.kv_store_factory = _fake_kv_store

    # Force training to fail so we enter the error handling path
    def _fail_container(settings: Settings) -> NoReturn:
        raise RuntimeError("training failed")

    _test_hooks.service_container_from_settings = _fail_container

    # The original training error should be re-raised (not the Redis error)
    # The Redis error is logged as a warning (line 678)
    with pytest.raises(RuntimeError, match="training failed"):
        train_job.process_train_job(payload)

    client.assert_only_called({"hset", "publish", "set"})
