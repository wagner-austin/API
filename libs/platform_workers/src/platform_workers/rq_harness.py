from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypedDict

from platform_core.logging import get_logger

from .redis import (
    _RedisBytesClient,
    redis_raw_for_rq,
)

log = get_logger(__name__)

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


class _RQModuleProtocol(Protocol):
    """Protocol for the rq module interface we use."""

    Queue: _QueueCtor
    SimpleWorker: _WorkerCtorRaw
    Retry: type[RQRetryLike]

    def get_current_job(self) -> CurrentJobProto | None: ...


def _load_rq_module() -> _RQModuleProtocol:
    """Load the rq module, using test hook if set."""
    from .testing import hooks

    if hooks.load_rq_module is not None:
        return hooks.load_rq_module()
    return __import__("rq")


def rq_retry(*, max_retries: int, intervals: list[int]) -> RQRetryLike:
    """Create an RQ Retry object with typed interface.

    Use this instead of `from rq import Retry`.
    """
    rq_mod = _load_rq_module()
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
    rq_mod = _load_rq_module()
    queue_ctor: _QueueCtor = rq_mod.Queue
    return queue_ctor(name, connection=connection)


def rq_queue(name: str, connection: _RedisBytesClient) -> RQClientQueue:
    """Create an RQ queue client with typed interface.

    The connection should be a raw Redis client from `redis_raw_for_rq()`.
    RQ needs access to all Redis queue operations which our RedisBytesProto
    adapter doesn't expose.
    """
    rq_mod = _load_rq_module()
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
    rq_mod = _load_rq_module()
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
    rq_mod = _load_rq_module()
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


class FetchedJobProto(Protocol):
    """Protocol for a fetched RQ job."""

    def get_id(self) -> str: ...

    def get_status(self) -> str: ...

    def return_value(self) -> _JsonValue: ...


class _JobFetchCallable(Protocol):
    """Protocol for Job.fetch class method callable."""

    def __call__(self, __job_id: str, *, connection: _RedisBytesClient) -> FetchedJobProto: ...


class _RQJobClassProto(Protocol):
    """Protocol for the RQ Job class with fetch method."""

    fetch: _JobFetchCallable


def load_no_such_job_error() -> type[Exception]:
    """Load the NoSuchJobError exception type for callers to catch.

    Use this when you need to handle job-not-found cases.
    Get the exception class, then use it in exception handling
    where rq_fetch_job is called.

    Returns:
        The NoSuchJobError exception class from rq.exceptions.
    """
    rq_exc = __import__("rq.exceptions", fromlist=["NoSuchJobError"])
    exc_cls: type[Exception] = rq_exc.NoSuchJobError
    return exc_cls


def rq_fetch_job(job_id: str, connection: _RedisBytesClient) -> FetchedJobProto:
    """Fetch an RQ job by ID.

    Use this instead of `from rq.job import Job`.

    Args:
        job_id: The job UUID string.
        connection: Redis bytes client for RQ operations.

    Returns:
        The fetched job implementing FetchedJobProto.

    Raises:
        NoSuchJobError: If the job does not exist. Use load_no_such_job_error()
            to get the exception type for catching.
    """
    from .testing import hooks

    if hooks.fetch_job is not None:
        return hooks.fetch_job(job_id, connection)

    # Dynamic import with Protocol annotation to avoid Any types
    rq_job_mod = __import__("rq.job", fromlist=["Job"])
    job_cls: _RQJobClassProto = rq_job_mod.Job
    result: FetchedJobProto = job_cls.fetch(job_id, connection=connection)
    return result


__all__ = [
    "CurrentJobProto",
    "FetchedJobProto",
    "RQClientQueue",
    "RQJobLike",
    "RQRetryLike",
    "WorkerConfig",
    "_JobFetchCallable",
    "_QueueCtor",
    "_RQJobInternal",
    "_RQModuleProtocol",
    "_RQQueueInternal",
    "_RQWorkerInternal",
    "_RedisBytesClient",
    "_WorkerCtorRaw",
    "get_current_job",
    "load_no_such_job_error",
    "redis_raw_for_rq",
    "rq_fetch_job",
    "rq_queue",
    "rq_retry",
    "run_rq_worker",
]
