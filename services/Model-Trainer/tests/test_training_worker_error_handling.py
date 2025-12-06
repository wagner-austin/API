"""Tests for training_worker.py error handling - covers lines 559 and 678."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

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


class _FakeRedisNonRedisErrorOnSet:
    """Fake Redis that raises a non-Redis error on set after status is set to running."""

    def __init__(self: _FakeRedisNonRedisErrorOnSet) -> None:
        self._kv: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._sets: dict[str, set[str]] = {}
        self._set_call_count = 0

    def ping(self: _FakeRedisNonRedisErrorOnSet, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self: _FakeRedisNonRedisErrorOnSet, key: str, value: str) -> bool | str:
        self._set_call_count += 1
        self._kv[key] = value
        return True

    def get(self: _FakeRedisNonRedisErrorOnSet, key: str) -> str | None:
        return self._kv.get(key)

    def hset(self: _FakeRedisNonRedisErrorOnSet, key: str, mapping: dict[str, str]) -> int:
        # Fail with non-Redis error when trying to record error status
        if mapping.get("status") == "failed":
            raise KeyError("not a redis error - simulating non-Redis failure")
        cur = self._hashes.setdefault(key, {})
        cur.update(mapping)
        return len(mapping)

    def hgetall(self: _FakeRedisNonRedisErrorOnSet, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def publish(self: _FakeRedisNonRedisErrorOnSet, channel: str, message: str) -> int:
        return 1

    def scard(self: _FakeRedisNonRedisErrorOnSet, key: str) -> int:
        return len(self._sets.get(key, set()))

    def sadd(self: _FakeRedisNonRedisErrorOnSet, key: str, member: str) -> int:
        s = self._sets.setdefault(key, set())
        before = len(s)
        s.add(member)
        return 1 if len(s) > before else 0

    def hget(self: _FakeRedisNonRedisErrorOnSet, key: str, field: str) -> str | None:
        return None

    def sismember(self: _FakeRedisNonRedisErrorOnSet, key: str, member: str) -> bool:
        return False

    def close(self: _FakeRedisNonRedisErrorOnSet) -> None:
        pass


def test_process_train_job_reraises_non_redis_error_on_handle_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    """Cover training_worker.py line 559 (non-Redis error re-raises during error handling).

    When training fails and then recording the error also fails with a non-Redis error,
    the non-Redis error should be re-raised (line 559).
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

    # Use our fake Redis that raises non-Redis error when recording error
    client = _FakeRedisNonRedisErrorOnSet()

    def _fake_redis(settings: Settings) -> _FakeRedisNonRedisErrorOnSet:
        return client

    monkeypatch.setattr(train_job, "redis_client", _fake_redis)

    # Force training to fail so we enter the error handling path (line 553)
    class _C:
        @staticmethod
        def from_settings(_: Settings) -> None:
            raise RuntimeError("training failed")

    monkeypatch.setattr(train_job, "ServiceContainer", _C)

    # The non-Redis error during error handling should be re-raised (line 559)
    with pytest.raises(KeyError, match="not a redis error"):
        train_job.process_train_job(payload)


class _FakeRedisRedisErrorOnSet:
    """Fake Redis that raises a Redis error when recording error status."""

    def __init__(self: _FakeRedisRedisErrorOnSet) -> None:
        self._kv: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._sets: dict[str, set[str]] = {}

    def ping(self: _FakeRedisRedisErrorOnSet, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self: _FakeRedisRedisErrorOnSet, key: str, value: str) -> bool | str:
        self._kv[key] = value
        return True

    def get(self: _FakeRedisRedisErrorOnSet, key: str) -> str | None:
        return self._kv.get(key)

    def hset(self: _FakeRedisRedisErrorOnSet, key: str, mapping: dict[str, str]) -> int:
        # Fail with Redis error when trying to record error status
        if mapping.get("status") == "failed":
            raise RedisConnectionError("simulated redis connection failure")
        cur = self._hashes.setdefault(key, {})
        cur.update(mapping)
        return len(mapping)

    def hgetall(self: _FakeRedisRedisErrorOnSet, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def publish(self: _FakeRedisRedisErrorOnSet, channel: str, message: str) -> int:
        return 1

    def scard(self: _FakeRedisRedisErrorOnSet, key: str) -> int:
        return len(self._sets.get(key, set()))

    def sadd(self: _FakeRedisRedisErrorOnSet, key: str, member: str) -> int:
        s = self._sets.setdefault(key, set())
        before = len(s)
        s.add(member)
        return 1 if len(s) > before else 0

    def hget(self: _FakeRedisRedisErrorOnSet, key: str, field: str) -> str | None:
        return None

    def sismember(self: _FakeRedisRedisErrorOnSet, key: str, member: str) -> bool:
        return False

    def close(self: _FakeRedisRedisErrorOnSet) -> None:
        pass


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

    # Use our fake Redis that raises Redis error when recording error
    client = _FakeRedisRedisErrorOnSet()

    def _fake_redis(settings: Settings) -> _FakeRedisRedisErrorOnSet:
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
