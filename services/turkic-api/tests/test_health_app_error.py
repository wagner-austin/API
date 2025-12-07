"""Tests for health check endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.health import HealthResponse, ReadyResponse
from platform_workers.testing import FakeRedis, FakeRedisError, FakeRedisNonRedisError

from turkic_api.api.health import healthz_endpoint, readyz_endpoint


def test_healthz_always_returns_ok() -> None:
    """Test healthz_endpoint always returns status ok."""
    result: HealthResponse = healthz_endpoint()
    assert result == {"status": "ok"}


def test_readyz_success(tmp_path: Path) -> None:
    """Test readyz_endpoint returns ready when Redis, workers, and volume are healthy."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    redis = FakeRedis()
    redis.sadd("rq:workers", "worker-1")

    result: ReadyResponse = readyz_endpoint(redis=redis, data_dir=str(data_dir))
    assert result == {"status": "ready", "reason": None}
    redis.assert_only_called({"sadd", "ping", "scard"})


def test_readyz_redis_error_returns_degraded(tmp_path: Path) -> None:
    """Test readyz_endpoint returns degraded when Redis fails."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    redis = FakeRedisError()

    result: ReadyResponse = readyz_endpoint(redis=redis, data_dir=str(data_dir))
    assert result["status"] == "degraded"
    assert result["reason"] == "redis error"
    redis.assert_only_called({"ping"})


def test_readyz_volume_missing_returns_degraded(tmp_path: Path) -> None:
    """Test readyz_endpoint returns degraded when data volume is missing."""
    redis = FakeRedis()
    redis.sadd("rq:workers", "worker-1")
    missing_dir = str(tmp_path / "nonexistent")

    result: ReadyResponse = readyz_endpoint(redis=redis, data_dir=missing_dir)
    assert result["status"] == "degraded"
    assert result["reason"] == "data volume not found"
    redis.assert_only_called({"sadd", "ping", "scard"})


def test_readyz_non_redis_error_reraises(tmp_path: Path) -> None:
    """Test readyz_endpoint re-raises non-Redis exceptions."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    redis = FakeRedisNonRedisError()

    with pytest.raises(RuntimeError, match="simulated non-Redis failure"):
        readyz_endpoint(redis=redis, data_dir=str(data_dir))
    redis.assert_only_called({"ping"})
