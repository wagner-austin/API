"""Tests for platform_workers.health module."""

from __future__ import annotations

import pytest
from platform_core.health import ReadyResponse

from platform_workers.health import (
    logged_readyz_with_workers,
    readyz_redis,
    readyz_redis_with_workers,
)
from platform_workers.redis import RedisStrProto
from platform_workers.testing import (
    FakeLogger,
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


class TestLoggedReadyzWithWorkers:
    """The probe plus its logging, which two trainer services each wrote out.

    Driven with this package's own FakeLogger rather than a stdlib Logger and
    a handler: it records `extra` as a typed field, and `extra` is the point
    of logging this at all -- an operator finds a degraded service by its
    reason, not by the message text.
    """

    def test_a_ready_pool_returns_ready_and_logs_it(self) -> None:
        redis = FakeRedis()
        redis.sadd("rq:workers", "w1")
        logger = FakeLogger()

        result = logged_readyz_with_workers(redis, logger)

        assert result == {"status": "ready", "reason": None}
        assert [record.msg for record in logger.records] == ["readyz"]

    def test_a_degraded_pool_returns_the_reason_and_logs_it(self) -> None:
        logger = FakeLogger()

        result = logged_readyz_with_workers(FakeRedisNoPong(), logger)

        assert result == {"status": "degraded", "reason": "redis no-pong"}
        assert [record.msg for record in logger.records] == ["readyz degraded"]

    def test_the_degraded_record_carries_the_reason(self) -> None:
        logger = FakeLogger()

        logged_readyz_with_workers(FakeRedisNoPong(), logger)

        assert logger.records[0].extra == {
            "category": "api",
            "service": "health",
            "event": "readyz",
            "reason": "redis no-pong",
        }

    def test_the_ready_record_says_so_rather_than_carrying_a_reason(self) -> None:
        redis = FakeRedis()
        redis.sadd("rq:workers", "w1")
        logger = FakeLogger()

        logged_readyz_with_workers(redis, logger)

        assert logger.records[0].extra == {
            "category": "api",
            "service": "health",
            "event": "readyz",
            "status": "ready",
        }

    def test_logging_never_changes_the_verdict(self) -> None:
        """A readiness answer that depended on whether anyone was listening
        would be the wrong shape entirely, so the logged result must equal
        what the underlying probe decided on its own."""
        assert logged_readyz_with_workers(FakeRedis(), FakeLogger()) == readyz_redis_with_workers(
            FakeRedis()
        )

    def test_a_custom_workers_key_is_passed_through(self) -> None:
        redis = FakeRedis()
        redis.sadd("custom:workers", "w1")

        assert logged_readyz_with_workers(redis, FakeLogger(), workers_key="custom:workers") == {
            "status": "ready",
            "reason": None,
        }
