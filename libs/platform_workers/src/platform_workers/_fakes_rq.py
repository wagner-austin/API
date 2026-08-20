"""_fakes: FakeJob and related definitions."""

from __future__ import annotations

from typing import NamedTuple, Protocol

from platform_workers.rq_harness import _JsonValue

from .rq_harness import (
    FetchedJobProto,
    RQJobLike,
    RQRetryLike,
)


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
    job_id: str


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
                job_id=self._job_id,
            )
        )
        return FakeJob(self._job_id)

    def remove(self, job_or_id: str) -> int:
        """Drop one pending job with the given id, as RQ's LREM does.

        Exactly one entry is removed even when several share an id, because
        the real queue is a Redis list and ``remove`` issues ``LREM key 1 id``.
        A fake that cleared every match would let a test pass against
        behaviour the queue does not have.

        Args:
            job_or_id: The job id to remove.

        Returns:
            1 when a pending job was removed, 0 when none matched.
        """
        for index, job in enumerate(self.jobs):
            if job.job_id == job_or_id:
                del self.jobs[index]
                return 1
        return 0


# =============================================================================
# Logger Fakes
# =============================================================================
