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


class FakeRedis(RedisStrProto):
    """In-memory Redis stub implementing RedisStrProto.

    Stores data in dictionaries for testing. Tracks published messages
    in `published` list for assertions.
    """

    def __init__(self) -> None:
        self.published: list[Published] = []
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._sets: dict[str, set[str]] = {}

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self, key: str, value: str) -> bool:
        self._strings[key] = value
        return True

    def get(self, key: str) -> str | None:
        return self._strings.get(key)

    def delete(self, key: str) -> int:
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
        # FakeRedis doesn't actually implement TTL, just return True if key exists
        return key in self._strings or key in self._hashes or key in self._sets

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        bucket = self._hashes.setdefault(key, {})
        bucket.update(mapping)
        return len(mapping)

    def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def publish(self, channel: str, message: str) -> int:
        self.published.append(Published(channel, message))
        return 1

    def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    def sadd(self, key: str, member: str) -> int:
        bucket = self._sets.setdefault(key, set())
        before = len(bucket)
        bucket.add(member)
        return 1 if len(bucket) > before else 0

    def sismember(self, key: str, member: str) -> bool:
        return member in self._sets.get(key, set())

    def close(self) -> None:
        self._strings.clear()
        self._hashes.clear()
        self._sets.clear()


class FakeRedisBytesClient(RedisBytesProto):
    """In-memory Redis bytes stub implementing RedisBytesProto.

    Minimal implementation for testing RQ-related code that needs
    a bytes-mode Redis client.
    """

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def close(self) -> None:
        pass


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
    "FakeRetry",
    "LogRecord",
    "LoggerProtocol",
    "Published",
]
