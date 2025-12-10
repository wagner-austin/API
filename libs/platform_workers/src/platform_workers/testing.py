"""Testing utilities for platform_workers.

This module provides typed stubs for testing services that use platform_workers
infrastructure. These stubs implement the public protocols with in-memory storage.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import NamedTuple, Protocol

from .redis import RedisBytesProto, RedisStrProto
from .rq_harness import RQJobLike, RQRetryLike

# Recursive JSON type matching rq_harness._JsonValue
_JsonValue = dict[str, "_JsonValue"] | list["_JsonValue"] | str | int | float | bool | None


class Published(NamedTuple):
    """Record of a published message."""

    channel: str
    payload: str


class MethodCall(NamedTuple):
    """Record of a method call on FakeRedis."""

    method: str
    args: tuple[str | int | dict[str, str], ...]


class FakeRedis(RedisStrProto):
    """In-memory Redis stub implementing RedisStrProto.

    Stores data in dictionaries for testing. Tracks published messages
    in `published` list for assertions. The `closed` attribute tracks
    whether close() was called.

    Call tracking:
        All method calls are recorded in `calls` list. Use `assert_only_called()`
        to verify that only expected methods were called during a test.

    Example:
        redis = FakeRedis()
        # ... run code that uses redis ...
        redis.assert_only_called({"ping", "scard"})  # Fails if other methods called
    """

    def __init__(self) -> None:
        self.published: list[Published] = []
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._sets: dict[str, set[str]] = {}
        self.closed: bool = False
        self.calls: list[MethodCall] = []

    def _record(self, method: str, *args: str | int | dict[str, str]) -> None:
        """Record a method call for tracking."""
        self.calls.append(MethodCall(method, args))

    def assert_only_called(self, expected: set[str]) -> None:
        """Assert that only the expected methods were called.

        Args:
            expected: Set of method names that are allowed to have been called.

        Raises:
            AssertionError: If any method not in `expected` was called.
        """
        actual = {call.method for call in self.calls}
        unexpected = actual - expected
        if unexpected:
            raise AssertionError(
                f"Unexpected methods called: {unexpected}. "
                f"Expected only: {expected}. Actual calls: {actual}"
            )

    def get_calls(self, method: str) -> list[MethodCall]:
        """Get all calls to a specific method."""
        return [c for c in self.calls if c.method == method]

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        return True

    def set(self, key: str, value: str) -> bool:
        self._record("set", key, value)
        self._strings[key] = value
        return True

    def get(self, key: str) -> str | None:
        self._record("get", key)
        return self._strings.get(key)

    def delete(self, key: str) -> int:
        self._record("delete", key)
        if key in self._strings:
            del self._strings[key]
            return 1
        if key in self._hashes:
            del self._hashes[key]
            return 1
        if key in self._sets:
            del self._sets[key]
            return 1
        return 0

    def expire(self, key: str, time: int) -> bool:
        self._record("expire", key, time)
        # FakeRedis doesn't actually implement TTL, just return True if key exists
        return key in self._strings or key in self._hashes or key in self._sets

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._record("hset", key, mapping)
        bucket = self._hashes.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    def hget(self, key: str, field: str) -> str | None:
        self._record("hget", key, field)
        return self._hashes.get(key, {}).get(field)

    def hgetall(self, key: str) -> dict[str, str]:
        self._record("hgetall", key)
        return dict(self._hashes.get(key, {}))

    def publish(self, channel: str, message: str) -> int:
        self._record("publish", channel, message)
        self.published.append(Published(channel, message))
        return 1

    def scard(self, key: str) -> int:
        self._record("scard", key)
        return len(self._sets.get(key, set()))

    def sadd(self, key: str, member: str) -> int:
        self._record("sadd", key, member)
        bucket = self._sets.setdefault(key, set())
        before = len(bucket)
        bucket.add(member)
        return 1 if len(bucket) > before else 0

    def sismember(self, key: str, member: str) -> bool:
        self._record("sismember", key, member)
        return member in self._sets.get(key, set())

    def close(self) -> None:
        self._record("close")
        self.closed = True
        self._strings.clear()
        self._hashes.clear()
        self._sets.clear()


class FakeRedisNoPong(FakeRedis):
    """FakeRedis that returns False on ping (simulates unhealthy Redis).

    Use this to test health check degraded paths when Redis is unreachable
    but doesn't raise an exception.

    Example:
        redis = FakeRedisNoPong()
        assert redis.ping() is False
        redis.assert_only_called({"ping"})
    """

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        return False


class FakeRedisError(FakeRedis):
    """FakeRedis that raises RedisError on ping (simulates Redis failure).

    Use this to test health check error handling paths when Redis raises
    an exception.

    Example:
        redis = FakeRedisError()
        with pytest.raises(RedisError):
            redis.ping()
        redis.assert_only_called({"ping"})
    """

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        from .redis import _load_redis_error_class

        error_cls = _load_redis_error_class()
        raise error_cls("simulated Redis failure")


class FakeRedisNonRedisError(FakeRedis):
    """FakeRedis that raises a non-Redis error on ping.

    Use this to test that non-Redis exceptions are properly propagated
    rather than being caught as Redis errors.

    Example:
        redis = FakeRedisNonRedisError()
        with pytest.raises(RuntimeError):
            redis.ping()
    """

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        raise RuntimeError("simulated non-Redis failure")


class FakeRedisPublishError(FakeRedis):
    """FakeRedis that raises on publish (simulates publish failure).

    Use this to test error handling when Redis publish operations fail.

    Example:
        redis = FakeRedisPublishError()
        with pytest.raises(OSError):
            redis.publish("channel", "message")
    """

    def publish(self, channel: str, message: str) -> int:
        self._record("publish", channel, message)
        raise OSError("simulated publish failure")


class FakeRedisScardError(FakeRedis):
    """FakeRedis that raises RedisError on scard (simulates worker check failure).

    Use this to test health check error handling when scard() fails but ping succeeds.

    Example:
        redis = FakeRedisScardError()
        redis.ping()  # OK
        with pytest.raises(RedisError):
            redis.scard("rq:workers")
    """

    def scard(self, key: str) -> int:
        self._record("scard", key)
        from .redis import _load_redis_error_class

        error_cls = _load_redis_error_class()
        raise error_cls("simulated scard failure")


class FakeRedisNonRedisScardError(FakeRedis):
    """FakeRedis that raises non-Redis error on scard.

    Use this to test that non-Redis exceptions on scard are properly propagated.

    Example:
        redis = FakeRedisNonRedisScardError()
        redis.ping()  # OK
        with pytest.raises(TypeError):
            redis.scard("rq:workers")
    """

    def scard(self, key: str) -> int:
        self._record("scard", key)
        raise TypeError("simulated non-Redis scard failure")


class FakeRedisHsetError(FakeRedis):
    """FakeRedis that raises non-Redis error on hset.

    Use this to test error handling when recording job status fails
    with a non-Redis error that should be re-raised.

    Example:
        redis = FakeRedisHsetError()
        with pytest.raises(RuntimeError):
            redis.hset("job:123", {"status": "failed"})
    """

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._record("hset", key, mapping)
        raise RuntimeError("simulated hset failure")


class FakeRedisHsetRedisError(FakeRedis):
    """FakeRedis that raises RedisError on hset.

    Use this to test error handling when recording job status fails
    with a Redis error that should be logged and swallowed.

    Example:
        redis = FakeRedisHsetRedisError()
        with pytest.raises(RedisError):
            redis.hset("job:123", {"status": "failed"})
    """

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._record("hset", key, mapping)
        from .redis import _load_redis_error_class

        raise _load_redis_error_class()("simulated Redis hset failure")


class FakeRedisBytesClient(RedisBytesProto):
    """In-memory Redis bytes stub implementing RedisBytesProto.

    Minimal implementation for testing RQ-related code that needs
    a bytes-mode Redis client. Tracks whether close() was called.
    """

    def __init__(self) -> None:
        self._closed = False

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        """Check if close() was called."""
        return self._closed


class FakeRedisClient:
    """In-memory Redis stub matching internal _RedisStrClient protocol.

    This class matches the raw redis.Redis[str] interface used by _RedisStrAdapter.
    Use this for testing internal adapter code. For testing public APIs,
    use FakeRedis instead which implements RedisStrProto.

    Call tracking:
        All method calls are recorded in `calls` list. Use `assert_only_called()`
        to verify that only expected methods were called during a test.
    """

    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._sets: dict[str, set[str]] = {}
        self.calls: list[MethodCall] = []

    def _record(self, method: str, *args: str | int | dict[str, str]) -> None:
        """Record a method call for tracking."""
        self.calls.append(MethodCall(method, args))

    def assert_only_called(self, expected: set[str]) -> None:
        """Assert that only the expected methods were called."""
        actual = {call.method for call in self.calls}
        unexpected = actual - expected
        if unexpected:
            raise AssertionError(
                f"Unexpected methods called: {unexpected}. "
                f"Expected only: {expected}. Actual calls: {actual}"
            )

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        return True

    def set(self, name: str, value: str) -> bool:
        self._record("set", name, value)
        self._strings[name] = value
        return True

    def get(self, name: str) -> str | None:
        self._record("get", name)
        return self._strings.get(name)

    def delete(self, *names: str) -> int:
        self._record("delete", *names)
        removed = 0
        for name in names:
            if name in self._strings:
                del self._strings[name]
                removed += 1
            if name in self._hashes:
                del self._hashes[name]
                removed += 1
            if name in self._sets:
                del self._sets[name]
                removed += 1
        return removed

    def expire(self, name: str, time: int) -> bool:
        self._record("expire", name, time)
        return name in self._strings or name in self._hashes or name in self._sets

    def hset(self, name: str, mapping: dict[str, str]) -> int:
        self._record("hset", name, mapping)
        bucket = self._hashes.setdefault(name, {})
        bucket.update(mapping)
        return len(mapping)

    def hget(self, name: str, key: str) -> str | None:
        self._record("hget", name, key)
        return self._hashes.get(name, {}).get(key)

    def hgetall(self, name: str) -> dict[str, str]:
        self._record("hgetall", name)
        return dict(self._hashes.get(name, {}))

    def publish(self, channel: str, message: str) -> int:
        self._record("publish", channel, message)
        return 1

    def scard(self, name: str) -> int:
        self._record("scard", name)
        return len(self._sets.get(name, set()))

    def sadd(self, name: str, value: str) -> int:
        self._record("sadd", name, value)
        bucket = self._sets.setdefault(name, set())
        before = len(bucket)
        bucket.add(value)
        return 1 if len(bucket) > before else 0

    def sismember(self, name: str, value: str) -> bool:
        self._record("sismember", name, value)
        return value in self._sets.get(name, set())

    def close(self) -> None:
        self._record("close")


class FakeJob(RQJobLike):
    """Fake RQ job for testing."""

    def __init__(self, job_id: str = "test-job-id") -> None:
        self._id = job_id

    def get_id(self) -> str:
        return self._id


class FakeRetry(RQRetryLike):
    """Fake RQ Retry for testing.

    Stores retry configuration without connecting to RQ.
    """

    def __init__(self, *, max: int, interval: list[int]) -> None:
        self.max_retries = max
        self.intervals = interval


class _EnqCallable(Protocol):
    """Protocol for callable that can be enqueued."""

    def __call__(
        self,
        *args: _JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike: ...


class EnqueuedJob(NamedTuple):
    """Record of an enqueued job."""

    func: str
    args: tuple[_JsonValue, ...]
    job_timeout: int | None
    result_ttl: int | None
    failure_ttl: int | None
    description: str | None


class FakeQueue:
    """Fake job queue for testing.

    Tracks enqueued jobs in `jobs` list for assertions.
    """

    def __init__(self, job_id: str = "test-job-id") -> None:
        self._job_id = job_id
        self.jobs: list[EnqueuedJob] = []

    def enqueue(
        self,
        func: str | _EnqCallable,
        *args: _JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        func_name = func if isinstance(func, str) else str(func)
        self.jobs.append(
            EnqueuedJob(
                func=func_name,
                args=args,
                job_timeout=job_timeout,
                result_ttl=result_ttl,
                failure_ttl=failure_ttl,
                description=description,
            )
        )
        return FakeJob(self._job_id)


class LoggerProtocol(Protocol):
    """Protocol for a minimal structured logger interface."""

    def debug(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None: ...

    def info(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None: ...

    def warning(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None: ...

    def error(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None: ...


class LogRecord(NamedTuple):
    """Record of a log message."""

    level: str
    msg: str
    args: tuple[_JsonValue, ...]
    extra: Mapping[str, _JsonValue] | None


class FakeLogger:
    """Fake logger for testing.

    Tracks log records in `records` list for assertions.
    """

    def __init__(self) -> None:
        self.records: list[LogRecord] = []

    def debug(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None:
        self.records.append(LogRecord("debug", msg, args, extra))

    def info(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None:
        self.records.append(LogRecord("info", msg, args, extra))

    def warning(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None:
        self.records.append(LogRecord("warning", msg, args, extra))

    def error(
        self,
        msg: str,
        *args: _JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, _JsonValue] | None = None,
    ) -> None:
        self.records.append(LogRecord("error", msg, args, extra))


__all__ = [
    "EnqueuedJob",
    "FakeJob",
    "FakeLogger",
    "FakeQueue",
    "FakeRedis",
    "FakeRedisBytesClient",
    "FakeRedisClient",
    "FakeRedisError",
    "FakeRedisHsetError",
    "FakeRedisHsetRedisError",
    "FakeRedisNoPong",
    "FakeRedisNonRedisError",
    "FakeRedisNonRedisScardError",
    "FakeRedisPublishError",
    "FakeRedisScardError",
    "FakeRetry",
    "LogRecord",
    "LoggerProtocol",
    "MethodCall",
    "Published",
]
