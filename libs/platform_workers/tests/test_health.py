"""Tests for platform_workers.health module."""

from __future__ import annotations

import pytest
from platform_core.health import ReadyResponse

from platform_workers.health import readyz_redis, readyz_redis_with_workers
from platform_workers.redis import RedisStrProto
from platform_workers.testing import (
    FakeRedis,
    FakeRedisError,
    FakeRedisNonRedisError,
    FakeRedisNonRedisScardError,
    FakeRedisNoPong,
    FakeRedisScardError,
)


def test_readyz_redis_healthy() -> None:
    """Test readyz_redis returns ready when Redis is healthy."""
    redis = FakeRedis()
    result: ReadyResponse = readyz_redis(redis)
    assert result == {"status": "ready", "reason": None}
    redis.assert_only_called({"ping"})


def test_readyz_redis_no_pong() -> None:
    """Test readyz_redis returns degraded when ping returns False."""
    redis = FakeRedisNoPong()
    result: ReadyResponse = readyz_redis(redis)
    assert result == {"status": "degraded", "reason": "redis no-pong"}
    redis.assert_only_called({"ping"})


def test_readyz_redis_error() -> None:
    """Test readyz_redis returns degraded when Redis raises error."""
    redis = FakeRedisError()
    result: ReadyResponse = readyz_redis(redis)
    assert result == {"status": "degraded", "reason": "redis error"}
    redis.assert_only_called({"ping"})


def test_readyz_redis_non_redis_error_raises() -> None:
    """Test readyz_redis re-raises non-Redis errors."""
    redis: RedisStrProto = FakeRedisNonRedisError()
    with pytest.raises(RuntimeError, match="simulated non-Redis failure"):
        readyz_redis(redis)


def test_readyz_redis_with_workers_healthy() -> None:
    """Test readyz_redis_with_workers returns ready when workers present."""
    redis = FakeRedis()
    redis.sadd("rq:workers", "worker-1")
    redis.calls.clear()  # Clear setup calls before testing
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "ready", "reason": None}
    redis.assert_only_called({"ping", "scard"})


def test_readyz_redis_with_workers_no_workers() -> None:
    """Test readyz_redis_with_workers returns degraded when no workers."""
    redis = FakeRedis()
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "no-worker"}
    redis.assert_only_called({"ping", "scard"})


def test_readyz_redis_with_workers_no_pong() -> None:
    """Test readyz_redis_with_workers returns degraded when ping fails."""
    redis = FakeRedisNoPong()
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "redis no-pong"}
    redis.assert_only_called({"ping"})


def test_readyz_redis_with_workers_ping_error() -> None:
    """Test readyz_redis_with_workers returns degraded on ping error."""
    redis = FakeRedisError()
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "redis error"}
    redis.assert_only_called({"ping"})


def test_readyz_redis_with_workers_scard_error() -> None:
    """Test readyz_redis_with_workers returns degraded on scard error."""
    redis = FakeRedisScardError()
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "redis error"}
    redis.assert_only_called({"ping", "scard"})


def test_readyz_redis_with_workers_non_redis_ping_error_raises() -> None:
    """Test readyz_redis_with_workers re-raises non-Redis ping errors."""
    redis: RedisStrProto = FakeRedisNonRedisError()
    with pytest.raises(RuntimeError, match="simulated non-Redis failure"):
        readyz_redis_with_workers(redis)


def test_readyz_redis_with_workers_non_redis_scard_error_raises() -> None:
    """Test readyz_redis_with_workers re-raises non-Redis scard errors."""
    redis: RedisStrProto = FakeRedisNonRedisScardError()
    with pytest.raises(TypeError, match="simulated non-Redis scard failure"):
        readyz_redis_with_workers(redis)


def test_readyz_redis_with_workers_custom_key() -> None:
    """Test readyz_redis_with_workers with custom workers_key."""
    redis = FakeRedis()
    redis.sadd("custom:workers", "w1")
    redis.calls.clear()  # Clear setup calls

    # Default key should show no workers
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "no-worker"}
    redis.assert_only_called({"ping", "scard"})

    redis.calls.clear()  # Clear for next test

    # Custom key should find worker
    result = readyz_redis_with_workers(redis, workers_key="custom:workers")
    assert result == {"status": "ready", "reason": None}
    redis.assert_only_called({"ping", "scard"})
