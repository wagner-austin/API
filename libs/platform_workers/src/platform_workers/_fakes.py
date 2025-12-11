"""Internal fake implementations for platform_workers testing.

This module contains all fake/stub implementations used for testing.
Import from testing.py for the public interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import NamedTuple, Protocol

from .redis import (
    PubSubMessage,
    RedisAsyncProto,
    RedisBytesProto,
    RedisPubSubProto,
    RedisStrProto,
    _RedisBytesClient,
)
from .rq_harness import (
    CurrentJobProto,
    FetchedJobProto,
    RQJobLike,
    RQRetryLike,
    _QueueCtor,
    _RQJobInternal,
    _RQQueueInternal,
    _WorkerCtorRaw,
)

# Recursive JSON type matching rq_harness._JsonValue
_JsonValue = dict[str, "_JsonValue"] | list["_JsonValue"] | str | int | float | bool | None


# =============================================================================
# Basic Types
# =============================================================================


class Published(NamedTuple):
    """Record of a published message."""

    channel: str
    payload: str


class MethodCall(NamedTuple):
    """Record of a method call on FakeRedis."""

    method: str
    args: tuple[str | int | dict[str, str], ...]


# =============================================================================
# Redis String Fakes
# =============================================================================


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
    """FakeRedis that returns False on ping (simulates unhealthy Redis)."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        return False


class FakeRedisError(FakeRedis):
    """FakeRedis that raises RedisError on ping (simulates Redis failure)."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        from .redis import _load_redis_error_class

        error_cls = _load_redis_error_class()
        raise error_cls("simulated Redis failure")


class FakeRedisNonRedisError(FakeRedis):
    """FakeRedis that raises a non-Redis error on ping."""

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        self._record("ping")
        raise RuntimeError("simulated non-Redis failure")


class FakeRedisPublishError(FakeRedis):
    """FakeRedis that raises on publish (simulates publish failure)."""

    def publish(self, channel: str, message: str) -> int:
        self._record("publish", channel, message)
        raise OSError("simulated publish failure")


class FakeRedisScardError(FakeRedis):
    """FakeRedis that raises RedisError on scard."""

    def scard(self, key: str) -> int:
        self._record("scard", key)
        from .redis import _load_redis_error_class

        error_cls = _load_redis_error_class()
        raise error_cls("simulated scard failure")


class FakeRedisNonRedisScardError(FakeRedis):
    """FakeRedis that raises non-Redis error on scard."""

    def scard(self, key: str) -> int:
        self._record("scard", key)
        raise TypeError("simulated non-Redis scard failure")


class FakeRedisHsetError(FakeRedis):
    """FakeRedis that raises non-Redis error on hset."""

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._record("hset", key, mapping)
        raise RuntimeError("simulated hset failure")


class FakeRedisHsetRedisError(FakeRedis):
    """FakeRedis that raises RedisError on hset."""

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._record("hset", key, mapping)
        from .redis import _load_redis_error_class

        raise _load_redis_error_class()("simulated Redis hset failure")


class FakeRedisConditionalHsetError(FakeRedis):
    """FakeRedis that raises RuntimeError on hset when status matches target."""

    def __init__(self, fail_on_status: str) -> None:
        super().__init__()
        self._fail_on_status = fail_on_status

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._record("hset", key, mapping)
        if mapping.get("status") == self._fail_on_status:
            raise RuntimeError("simulated hset failure on status=" + self._fail_on_status)
        for k, v in mapping.items():
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key][k] = v
        return len(mapping)


class FakeRedisConditionalHsetRedisError(FakeRedis):
    """FakeRedis that raises RedisError on hset when status matches target."""

    def __init__(self, fail_on_status: str) -> None:
        super().__init__()
        self._fail_on_status = fail_on_status

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._record("hset", key, mapping)
        if mapping.get("status") == self._fail_on_status:
            from .redis import _load_redis_error_class

            msg = "simulated Redis hset failure on status=" + self._fail_on_status
            raise _load_redis_error_class()(msg)
        for k, v in mapping.items():
            if key not in self._hashes:
                self._hashes[key] = {}
            self._hashes[key][k] = v
        return len(mapping)


# =============================================================================
# Redis Bytes Fakes
# =============================================================================


class FakeRedisBytesClient(RedisBytesProto):
    """In-memory Redis bytes stub implementing RedisBytesProto."""

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
    """In-memory Redis stub matching internal _RedisStrClient protocol."""

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


# =============================================================================
# RQ Job Fakes
# =============================================================================


class FakeJob(RQJobLike):
    """Fake RQ job for testing."""

    def __init__(self, job_id: str = "test-job-id") -> None:
        self._id = job_id

    def get_id(self) -> str:
        return self._id


class FakeFetchedJob(FetchedJobProto):
    """Fake fetched RQ job for testing get_job_status."""

    def __init__(
        self,
        job_id: str = "test-job-id",
        status: str = "finished",
        result: _JsonValue = None,
    ) -> None:
        self._id = job_id
        self._status = status
        self._result = result

    def get_id(self) -> str:
        return self._id

    def get_status(self) -> str:
        return self._status

    def return_value(self) -> _JsonValue:
        return self._result


class FakeRetry(RQRetryLike):
    """Fake RQ Retry for testing."""

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
    """Fake job queue for testing."""

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


# =============================================================================
# Logger Fakes
# =============================================================================


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
    """Fake logger for testing."""

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


# =============================================================================
# Async Redis Fakes for PubSub Testing
# =============================================================================


class FakePubSub(RedisPubSubProto):
    """Fake async Redis PubSub client for testing."""

    def __init__(self) -> None:
        self.subscriptions: list[str] = []
        self._messages: list[PubSubMessage] = []
        self._closed = False

    def inject_message(self, channel: str, data: str) -> None:
        """Inject a message to be returned by get_message."""
        msg: PubSubMessage = {"type": "message", "pattern": None, "channel": channel, "data": data}
        self._messages.append(msg)

    async def subscribe(self, *channels: str) -> None:
        self.subscriptions.extend(channels)

    async def get_message(
        self, *, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> PubSubMessage | None:
        if self._messages:
            return self._messages.pop(0)
        return None

    async def close(self) -> None:
        self._closed = True


class FakeAsyncRedis(RedisAsyncProto):
    """Fake async Redis client for PubSub testing."""

    def __init__(self) -> None:
        self._pubsub = FakePubSub()

    def pubsub(self) -> FakePubSub:
        return self._pubsub


# =============================================================================
# Redis Module Fakes for Runtime Import Testing
# =============================================================================


class FakeRedisStrModule:
    """Fake redis module for str client factory testing."""

    def __init__(self, client: FakeRedisClient) -> None:
        self._client = client
        self.from_url_called = False
        self.from_url_args: tuple[str, ...] = ()

    def from_url(
        self,
        url: str,
        *,
        encoding: str,
        decode_responses: bool,
        socket_connect_timeout: float,
        socket_timeout: float,
        retry_on_timeout: bool,
    ) -> FakeRedisClient:
        self.from_url_called = True
        self.from_url_args = (url, encoding)
        return self._client


class FakeRedisBytesModule:
    """Fake redis module for bytes client factory testing."""

    def __init__(self) -> None:
        self._client = _FakeBytesClientInternal()
        self.from_url_called = False
        self.from_url_url: str = ""

    def from_url(
        self,
        url: str,
        *,
        decode_responses: bool,
        socket_connect_timeout: float,
        socket_timeout: float,
        retry_on_timeout: bool,
    ) -> _FakeBytesClientInternal:
        self.from_url_called = True
        self.from_url_url = url
        return self._client


class _FakeBytesClientInternal:
    """Internal bytes client matching _RedisBytesClient protocol."""

    def __init__(self) -> None:
        self._closed = False

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def close(self) -> None:
        self._closed = True


class FakeRedisAsyncioModule:
    """Fake redis.asyncio module for pubsub testing."""

    def __init__(self) -> None:
        self._client = FakeAsyncRedis()
        self.from_url_called = False
        self.from_url_url: str = ""

    def from_url(self, url: str, *, encoding: str, decode_responses: bool) -> FakeAsyncRedis:
        self.from_url_called = True
        self.from_url_url = url
        return self._client


# =============================================================================
# RQ Module Fakes for Runtime Import Testing
# =============================================================================


class _FakeCurrentJob:
    """Fake RQ current job for testing get_current_job."""

    origin: str | None

    def __init__(self, job_id: str = "test-job-id", origin: str | None = "test-queue") -> None:
        self._id = job_id
        self.origin = origin

    def get_id(self) -> str:
        return self._id


class _FakeRQJob:
    """Internal fake RQ job for testing."""

    def __init__(self, job_id: str = "fake-job-id") -> None:
        self._id = job_id

    def get_id(self) -> str:
        return self._id


class _FakeRQQueueInternal:
    """Internal fake RQ queue matching _RQQueueInternal protocol."""

    def __init__(self, name: str, *, connection: _RedisBytesClient) -> None:
        self.name = name
        self.connection = connection
        self._job_id = "fake-job-id"

    def enqueue(
        self,
        func_ref: str,
        *args: _JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> _RQJobInternal:
        return _FakeRQJob(f"job-{func_ref}")


class _FakeRQWorkerInternal:
    """Internal fake RQ worker matching _RQWorkerInternal protocol."""

    def __init__(
        self,
        queues: list[_RQQueueInternal],
        *,
        connection: _RedisBytesClient,
    ) -> None:
        self.queues = queues
        self.connection = connection
        self.work_called = False
        self.with_scheduler: bool | None = None

    def work(self, *, with_scheduler: bool) -> None:
        self.work_called = True
        self.with_scheduler = with_scheduler


class FakeRQModule:
    """Fake rq module for testing without real RQ dependency."""

    Queue: _QueueCtor
    SimpleWorker: _WorkerCtorRaw
    Retry: type[RQRetryLike]

    def __init__(self, *, current_job: _FakeCurrentJob | None = None) -> None:
        self._current_job = current_job
        self.Queue = _FakeRQQueueInternal
        self.SimpleWorker = _FakeRQWorkerInternal
        self.Retry = FakeRetry

    def get_current_job(self) -> CurrentJobProto | None:
        return self._current_job


__all__ = [
    "EnqueuedJob",
    "FakeAsyncRedis",
    "FakeFetchedJob",
    "FakeJob",
    "FakeLogger",
    "FakePubSub",
    "FakeQueue",
    "FakeRQModule",
    "FakeRedis",
    "FakeRedisAsyncioModule",
    "FakeRedisBytesClient",
    "FakeRedisBytesModule",
    "FakeRedisClient",
    "FakeRedisConditionalHsetError",
    "FakeRedisConditionalHsetRedisError",
    "FakeRedisError",
    "FakeRedisHsetError",
    "FakeRedisHsetRedisError",
    "FakeRedisNoPong",
    "FakeRedisNonRedisError",
    "FakeRedisNonRedisScardError",
    "FakeRedisPublishError",
    "FakeRedisScardError",
    "FakeRedisStrModule",
    "FakeRetry",
    "LogRecord",
    "LoggerProtocol",
    "MethodCall",
    "Published",
    "_FakeCurrentJob",
    "_FakeRQQueueInternal",
    "_FakeRQWorkerInternal",
]
