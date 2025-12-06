"""Tests for platform_workers.health module."""

from __future__ import annotations

import pytest
from platform_core.health import ReadyResponse

from platform_workers.health import readyz_redis, readyz_redis_with_workers
from platform_workers.redis import RedisStrProto, _load_redis_error_class

# Get the actual redis.exceptions.RedisError for tests
_ActualRedisError = _load_redis_error_class()


class _FakeRedisBase:
    """Base fake Redis with all required Protocol methods stubbed."""

    def __init__(self) -> None:
        self._workers: set[str] = set()

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self, key: str, value: str) -> bool | str | None:
        return True

    def get(self, key: str) -> str | None:
        return None

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        return len(mapping)

    def hget(self, key: str, field: str) -> str | None:
        return None

    def hgetall(self, key: str) -> dict[str, str]:
        return {}

    def publish(self, channel: str, message: str) -> int:
        return 1

    def scard(self, key: str) -> int:
        return len(self._workers)

    def sadd(self, key: str, member: str) -> int:
        return 1

    def sismember(self, key: str, member: str) -> bool:
        return False

    def delete(self, key: str) -> int:
        return 0

    def expire(self, key: str, time: int) -> bool:
        return False

    def close(self) -> None:
        pass


class _FakeRedisHealthy(_FakeRedisBase):
    """Fake Redis client that is healthy."""

    def add_worker(self, worker_id: str) -> None:
        self._workers.add(worker_id)


class _FakeRedisNoPong(_FakeRedisBase):
    """Fake Redis client that returns False for ping."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return False


class _FakeRedisPingError(_FakeRedisBase):
    """Fake Redis client that raises RedisError on ping."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        raise _ActualRedisError("connection refused")

    def scard(self, key: str) -> int:
        raise _ActualRedisError("connection refused")


class _FakeRedisScardError(_FakeRedisBase):
    """Fake Redis client that pings OK but raises on scard."""

    def scard(self, key: str) -> int:
        raise _ActualRedisError("connection refused")


class _FakeRedisNonRedisError(_FakeRedisBase):
    """Fake Redis client that raises a non-Redis error."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        raise ValueError("unexpected error")

    def scard(self, key: str) -> int:
        raise ValueError("unexpected error")


def test_readyz_redis_healthy() -> None:
    """Test readyz_redis returns ready when Redis is healthy."""
    redis: RedisStrProto = _FakeRedisHealthy()
    result: ReadyResponse = readyz_redis(redis)
    assert result == {"status": "ready", "reason": None}


def test_readyz_redis_no_pong() -> None:
    """Test readyz_redis returns degraded when ping returns False."""
    redis: RedisStrProto = _FakeRedisNoPong()
    result: ReadyResponse = readyz_redis(redis)
    assert result == {"status": "degraded", "reason": "redis no-pong"}


def test_readyz_redis_error() -> None:
    """Test readyz_redis returns degraded when Redis raises error."""
    redis: RedisStrProto = _FakeRedisPingError()
    result: ReadyResponse = readyz_redis(redis)
    assert result == {"status": "degraded", "reason": "redis error"}


def test_readyz_redis_non_redis_error_raises() -> None:
    """Test readyz_redis re-raises non-Redis errors."""
    redis: RedisStrProto = _FakeRedisNonRedisError()
    with pytest.raises(ValueError, match="unexpected error"):
        readyz_redis(redis)


def test_readyz_redis_with_workers_healthy() -> None:
    """Test readyz_redis_with_workers returns ready when workers present."""
    redis = _FakeRedisHealthy()
    redis.add_worker("worker-1")
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "ready", "reason": None}


def test_readyz_redis_with_workers_no_workers() -> None:
    """Test readyz_redis_with_workers returns degraded when no workers."""
    redis: RedisStrProto = _FakeRedisHealthy()
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "no-worker"}


def test_readyz_redis_with_workers_no_pong() -> None:
    """Test readyz_redis_with_workers returns degraded when ping fails."""
    redis: RedisStrProto = _FakeRedisNoPong()
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "redis no-pong"}


def test_readyz_redis_with_workers_ping_error() -> None:
    """Test readyz_redis_with_workers returns degraded on ping error."""
    redis: RedisStrProto = _FakeRedisPingError()
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "redis error"}


def test_readyz_redis_with_workers_scard_error() -> None:
    """Test readyz_redis_with_workers returns degraded on scard error."""
    redis: RedisStrProto = _FakeRedisScardError()
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "redis error"}


def test_readyz_redis_with_workers_non_redis_ping_error_raises() -> None:
    """Test readyz_redis_with_workers re-raises non-Redis ping errors."""
    redis: RedisStrProto = _FakeRedisNonRedisError()
    with pytest.raises(ValueError, match="unexpected error"):
        readyz_redis_with_workers(redis)


class _FakeRedisNonRedisScardError(_FakeRedisBase):
    """Fake Redis client that pings OK but raises non-Redis error on scard."""

    def scard(self, key: str) -> int:
        raise TypeError("unexpected scard error")


def test_readyz_redis_with_workers_non_redis_scard_error_raises() -> None:
    """Test readyz_redis_with_workers re-raises non-Redis scard errors."""
    redis: RedisStrProto = _FakeRedisNonRedisScardError()
    with pytest.raises(TypeError, match="unexpected scard error"):
        readyz_redis_with_workers(redis)


class _FakeRedisCustomKey(_FakeRedisBase):
    """Fake Redis client with custom worker keys."""

    def __init__(self) -> None:
        super().__init__()
        self._sets: dict[str, set[str]] = {"custom:workers": {"w1"}}

    def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))


def test_readyz_redis_with_workers_custom_key() -> None:
    """Test readyz_redis_with_workers with custom workers_key."""
    redis: RedisStrProto = _FakeRedisCustomKey()
    # Default key should show no workers
    result: ReadyResponse = readyz_redis_with_workers(redis)
    assert result == {"status": "degraded", "reason": "no-worker"}

    # Custom key should find worker
    result = readyz_redis_with_workers(redis, workers_key="custom:workers")
    assert result == {"status": "ready", "reason": None}
