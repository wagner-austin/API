"""_fakes: FakeRedisStrModule and related definitions."""

from __future__ import annotations

from platform_workers._fakes_logging import FakeAsyncRedis
from platform_workers._fakes_redis import FakeRedisClient
from platform_workers._fakes_rq import FakeRetry
from platform_workers.rq_harness import _JsonValue

from .redis import (
    _RedisBytesClient,
)
from .rq_harness import (
    CurrentJobProto,
    RQRetryLike,
    _QueueCtor,
    _RQJobInternal,
    _RQQueueInternal,
    _WorkerCtorRaw,
)


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
    """Internal fake RQ job for testing.

    Mirrors rq's own Job, which exposes `id` as a property. This fake used to
    provide `get_id()` instead; because the harness called that, every test
    passed while the real rq raised AttributeError in production.
    """

    def __init__(self, job_id: str = "fake-job-id") -> None:
        self._id = job_id

    @property
    def id(self) -> str:
        """Job identifier.

        Returns:
            The job id, as rq's Job.id does.
        """
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
