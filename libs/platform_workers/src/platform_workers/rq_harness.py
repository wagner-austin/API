from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypedDict

from .redis import (
    _RedisBytesClient,
    redis_raw_for_rq,
)

# Recursive JSON type for queue payloads
_JsonValue = dict[str, "_JsonValue"] | list["_JsonValue"] | str | int | float | bool | None


class WorkerConfig(TypedDict):
    redis_url: str
    queue_name: str
    events_channel: str


# Internal protocols for RQ types
class _RQQueueLike(Protocol):
    """Protocol for RQ Queue-like objects."""

    ...


class _RQWorkerLike(Protocol):
    """Protocol for RQ Worker-like objects."""

    def work(self, *, with_scheduler: bool) -> None: ...


class _RQJobInternal(Protocol):
    """Protocol for internal RQ Job object."""

    def get_id(self) -> str: ...


class _RQQueueInternal(Protocol):
    """Protocol for internal RQ Queue."""

    def enqueue(
        self,
        func_ref: str,
        *args: _JsonValue,
        job_timeout: int | None = ...,
        result_ttl: int | None = ...,
        failure_ttl: int | None = ...,
        retry: RQRetryLike | None = ...,
        description: str | None = ...,
    ) -> _RQJobInternal: ...


class _RQWorkerInternal(Protocol):
    """Protocol for internal RQ SimpleWorker."""

    def work(self, *, with_scheduler: bool) -> None: ...


class RQRetryLike(Protocol):
    """Public protocol for RQ Retry object.

    Defines the constructor signature for RQ's Retry class.
    Use `rq_retry()` helper to create instances with proper typing.
    """

    def __init__(self, *, max: int, interval: list[int]) -> None: ...


def rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    """Create an RQ Retry object with typed interface.

    Use this instead of `from rq import Retry`.
    """
    rq_mod = __import__("rq")
    retry_cls: type[RQRetryLike] = rq_mod.Retry
    retry: RQRetryLike = retry_cls(max=max_retries, interval=intervals)
    return retry


class RQClientQueue(Protocol):
    """Public protocol for RQ queue client."""

    def enqueue(
        self,
        func_ref: str,
        *args: _JsonValue,
        job_timeout: int | None = ...,
        result_ttl: int | None = ...,
        failure_ttl: int | None = ...,
        retry: RQRetryLike | None = ...,
        description: str | None = ...,
    ) -> RQJobLike: ...


class RQJobLike(Protocol):
    """Public protocol for RQ Job."""

    def get_id(self) -> str: ...


class CurrentJobProto(Protocol):
    """Protocol for the current RQ job when running inside a worker."""

    origin: str | None

    def get_id(self) -> str: ...


class _QueueCtor(Protocol):
    def __call__(self, name: str, *, connection: _RedisBytesClient) -> _RQQueueInternal: ...


class _WorkerCtor(Protocol):
    def __call__(
        self, queues: list[_RQQueueLike], *, connection: _RedisBytesClient
    ) -> _RQWorkerInternal: ...


class _RQQueueAdapter(_RQQueueLike, RQClientQueue):
    def __init__(self, inner: _RQQueueInternal) -> None:
        self._inner = inner

    def enqueue(
        self,
        func_ref: str,
        *args: _JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike:
        job = self._inner.enqueue(
            func_ref,
            *args,
            job_timeout=job_timeout,
            result_ttl=result_ttl,
            failure_ttl=failure_ttl,
            retry=retry,
            description=description,
        )

        class _Job(RQJobLike):
            def __init__(self, inner: _RQJobInternal) -> None:
                self._inner = inner

            def get_id(self) -> str:
                return str(self._inner.get_id())

        return _Job(job)


def _rq_queue_raw(name: str, connection: _RedisBytesClient) -> _RQQueueInternal:
    """Create a raw RQ Queue for use with workers (not wrapped)."""
    rq_mod = __import__("rq")
    queue_ctor: _QueueCtor = rq_mod.Queue
    return queue_ctor(name, connection=connection)


def rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
    """Create an RQ queue client with typed interface.

    The connection should be a raw Redis client from `redis_raw_for_rq()`.
    RQ needs access to all Redis queue operations which our RedisBytesProto
    adapter doesn't expose.
    """
    rq_mod = __import__("rq")
    queue_ctor: _QueueCtor = rq_mod.Queue
    inner_q = queue_ctor(name, connection=connection)
    return _RQQueueAdapter(inner_q)


class _WorkerCtorRaw(Protocol):
    def __call__(
        self, queues: list[_RQQueueInternal], *, connection: _RedisBytesClient
    ) -> _RQWorkerInternal: ...


def _rq_simple_worker(
    queues: list[_RQQueueInternal], connection: _RedisBytesClient
) -> _RQWorkerLike:
    rq_mod = __import__("rq")
    worker_ctor: _WorkerCtorRaw = rq_mod.SimpleWorker
    worker = worker_ctor(queues, connection=connection)

    class _Worker(_RQWorkerLike):
        def __init__(self, inner: _RQWorkerInternal) -> None:
            self._inner = inner

        def work(self, *, with_scheduler: bool) -> None:
            self._inner.work(with_scheduler=with_scheduler)

    return _Worker(worker)


def get_current_job() -> CurrentJobProto | None:
    """Get the current RQ job when running inside a worker.

    Returns None if not running inside an RQ worker context.
    Use this instead of `from rq import get_current_job`.
    """
    rq_mod = __import__("rq")
    get_job_fn: Callable[[], CurrentJobProto | None] = rq_mod.get_current_job
    return get_job_fn()


def run_rq_worker(config: WorkerConfig) -> None:
    """Start an RQ worker bound to the configured queue and Redis connection.

    This function intentionally does not swallow exceptions or add fallback logic.
    It relies on the runtime RQ library; in tests, factories can be patched to
    provide strict fakes.
    """
    conn = redis_raw_for_rq(config["redis_url"])
    q = _rq_queue_raw(config["queue_name"], connection=conn)
    worker = _rq_simple_worker([q], connection=conn)
    worker.work(with_scheduler=True)


__all__ = [
    "CurrentJobProto",
    "RQClientQueue",
    "RQJobLike",
    "RQRetryLike",
    "WorkerConfig",
    "_RedisBytesClient",
    "get_current_job",
    "redis_raw_for_rq",
    "rq_queue",
    "rq_retry",
    "run_rq_worker",
]
