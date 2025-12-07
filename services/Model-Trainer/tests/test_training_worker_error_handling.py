"""Tests for training_worker.py error handling - covers lines 559 and 678."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from platform_workers.redis import _load_redis_error_class
from platform_workers.testing import FakeRedis

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


def test_process_train_job_reraises_non_redis_error_on_handle_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
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
    monkeypatch.setattr(train_job, "load_settings", lambda: settings)

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
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            assert fid == "deadbeef"
            return corpus_root

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    (corpus_root / "a.txt").write_text("hello\n", encoding="utf-8")

    # Use FakeRedis with conditional hset - only fail when status == "failed"
    # This allows the initial save (status=processing) to succeed, then fails
    # when _handle_train_error saves with status=failed
    client = FakeRedis()
    original_hset = client.hset

    def conditional_hset(key: str, mapping: dict[str, str]) -> int:
        if mapping.get("status") == "failed":
            client._record("hset", key, mapping)
            raise RuntimeError("non-redis error during error handling")
        return original_hset(key, mapping)

    monkeypatch.setattr(client, "hset", conditional_hset)

    def _fake_redis(settings: Settings) -> FakeRedis:
        return client

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)

    # Force training to fail so we enter the error handling path
    class _C:
        @staticmethod
        def from_settings(_: Settings) -> None:
            raise RuntimeError("training failed")

    monkeypatch.setattr(train_job, "ServiceContainer", _C)

    # The non-Redis error during error handling should be re-raised (line 421)
    with pytest.raises(RuntimeError, match="non-redis error during error handling"):
        train_job.process_train_job(payload)

    client.assert_only_called({"hset", "publish", "set"})


def test_process_train_job_logs_redis_error_on_handle_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
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
    monkeypatch.setattr(train_job, "load_settings", lambda: settings)

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
    from model_trainer.core.services.data import corpus_fetcher as cf

    class _CF:
        def __init__(self: _CF, api_url: str, api_key: str, cache_dir: Path) -> None:
            pass

        def fetch(self: _CF, fid: str) -> Path:
            assert fid == "deadbeef"
            return corpus_root

    monkeypatch.setattr(cf, "CorpusFetcher", _CF)
    (corpus_root / "a.txt").write_text("hello\n", encoding="utf-8")

    # Use FakeRedis with conditional hset - only fail when status == "failed"
    client = FakeRedis()
    original_hset = client.hset
    redis_error_cls = _load_redis_error_class()

    def conditional_hset(key: str, mapping: dict[str, str]) -> int:
        if mapping.get("status") == "failed":
            client._record("hset", key, mapping)
            raise redis_error_cls("simulated redis connection failure")
        return original_hset(key, mapping)

    monkeypatch.setattr(client, "hset", conditional_hset)

    def _fake_redis(settings: Settings) -> FakeRedis:
        return client

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)

    # Force training to fail so we enter the error handling path
    class _C:
        @staticmethod
        def from_settings(_: Settings) -> None:
            raise RuntimeError("training failed")

    monkeypatch.setattr(train_job, "ServiceContainer", _C)

    # The original training error should be re-raised (not the Redis error)
    # The Redis error is logged as a warning (line 678)
    with pytest.raises(RuntimeError, match="training failed"):
        train_job.process_train_job(payload)

    client.assert_only_called({"hset", "publish", "set"})
