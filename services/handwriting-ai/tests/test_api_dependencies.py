"""API dependency providers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from platform_core.config import _test_hooks as config_test_hooks
from platform_core.testing import make_fake_env
from platform_workers.rq_harness import _RedisBytesClient
from platform_workers.testing import FakeRedis, FakeRedisBytesClient

from handwriting_ai import _test_hooks
from handwriting_ai._hook_protocols import LoggerInstanceProtocol
from handwriting_ai.api.dependencies import (
    _get_redis_url,
    get_queue,
    get_redis,
    get_request_logger,
    get_settings,
)
from handwriting_ai.api.types import RQRetryLike, UnknownJson
from handwriting_ai.config import Settings


class _RedisConnectionProto(Protocol):
    """Protocol for Redis connection used by RQ."""

    pass


# --- Tests for dependencies.py ---


def test_get_settings_returns_settings(tmp_path: Path) -> None:
    """Test get_settings() returns loaded settings."""
    fake_settings: Settings = {
        "app": {
            "data_root": tmp_path / "data",
            "artifacts_root": tmp_path / "artifacts",
            "logs_root": tmp_path / "logs",
            "threads": 2,
            "port": 8080,
        },
        "digits": {
            "model_dir": tmp_path / "models",
            "active_model": "test-model",
            "tta": False,
            "uncertain_threshold": 0.5,
            "max_image_mb": 10,
            "max_image_side_px": 2048,
            "predict_timeout_seconds": 30,
            "visualize_max_kb": 128,
            "retention_keep_runs": 5,
            "allowed_hosts": frozenset(["*"]),
        },
        "security": {"api_key": "", "api_key_enabled": False},
    }

    def _fake_load_settings(*, create_dirs: bool = True) -> Settings:
        return fake_settings

    _test_hooks.load_settings = _fake_load_settings

    settings = get_settings()
    assert settings["app"]["threads"] == 2
    assert settings["app"]["port"] == 8080


def test_get_redis_url_returns_url() -> None:
    """Test _get_redis_url() returns REDIS_URL from environment."""
    config_test_hooks.get_env = make_fake_env({"REDIS_URL": "redis://localhost:6379/0"})
    url = _get_redis_url()
    assert url == "redis://localhost:6379/0"


def test_get_redis_url_raises_when_missing() -> None:
    """Test _get_redis_url() raises when REDIS_URL not set."""
    config_test_hooks.get_env = make_fake_env({})
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        _get_redis_url()


def test_get_redis_yields_and_closes() -> None:
    """Test get_redis() yields Redis client and closes on teardown."""
    config_test_hooks.get_env = make_fake_env({"REDIS_URL": "redis://localhost:6379/0"})

    redis_instance = FakeRedis()

    def _fake_redis_for_kv(url: str) -> FakeRedis:
        return redis_instance

    _test_hooks.redis_factory = _fake_redis_for_kv

    gen = get_redis()
    client = next(gen)
    if client is None:
        raise AssertionError("expected redis client")
    assert not redis_instance.closed

    # Exhaust the generator to trigger finally block using gen.close()
    gen.close()

    assert redis_instance.closed
    redis_instance.assert_only_called({"close"})


def test_get_request_logger_returns_logger() -> None:
    """Test get_request_logger() returns a logger instance."""

    class _FakeLoggerInstance:
        def info(
            self,
            msg: str,
            *args: float | int | str | Path | BaseException,
            extra: dict[str, str | int | float | bool | None] | None = None,
        ) -> None:
            pass

        def warning(
            self,
            msg: str,
            *args: float | int | str | Path | BaseException,
            extra: dict[str, str | int | float | bool | None] | None = None,
        ) -> None:
            pass

        def error(
            self,
            msg: str,
            *args: float | int | str | Path | BaseException,
            extra: dict[str, str | int | float | bool | None] | None = None,
        ) -> None:
            pass

        def debug(
            self,
            msg: str,
            *args: float | int | str | Path | BaseException,
            extra: dict[str, str | int | float | bool | None] | None = None,
        ) -> None:
            pass

    def _fake_get_logger(name: str) -> LoggerInstanceProtocol:
        return _FakeLoggerInstance()

    _test_hooks.get_logger = _fake_get_logger

    logger = get_request_logger()
    if logger is None:
        raise AssertionError("expected logger")


def test_get_queue_returns_queue_adapter() -> None:
    """Test get_queue() returns a QueueProtocol implementation."""
    config_test_hooks.get_env = make_fake_env({"REDIS_URL": "redis://localhost:6379/0"})

    enqueued: list[dict[str, UnknownJson]] = []

    class _FakeRQJob:
        def get_id(self) -> str:
            return "fake-job-id"

    class _FakeRQQueue:
        def enqueue(
            self,
            func: str,
            *args: UnknownJson,
            job_timeout: int | None = None,
            result_ttl: int | None = None,
            failure_ttl: int | None = None,
            retry: RQRetryLike | None = None,
            description: str | None = None,
        ) -> _FakeRQJob:
            enqueued.append({"func": func, "args": list(args)})
            return _FakeRQJob()

        def remove(self, job_or_id: str) -> int:
            """Report that nothing was pending; this fake records enqueues only.

            Args:
                job_or_id: The job id a caller would remove.

            Returns:
                0, since this double keeps no pending list to remove from.
            """
            _ = job_or_id
            return 0

    def _fake_rq_conn(url: str) -> FakeRedisBytesClient:
        return FakeRedisBytesClient()

    def _fake_rq_queue_factory(name: str, connection: _RedisBytesClient) -> _FakeRQQueue:
        return _FakeRQQueue()

    _test_hooks.rq_conn = _fake_rq_conn
    _test_hooks.rq_queue_factory = _fake_rq_queue_factory

    queue = get_queue()
    job = queue.enqueue("some.func", {"key": "value"}, job_timeout=60)
    assert job.get_id() == "fake-job-id"
    assert len(enqueued) == 1
    assert enqueued[0]["func"] == "some.func"


# --- Tests for training.py _validate_train_request ---
