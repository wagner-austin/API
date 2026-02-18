"""RQ adapter for job queue operations.

This module provides an adapter for RQ job queue operations.
"""

from __future__ import annotations

from typing import NamedTuple

from platform_core.json_utils import JSONObject
from platform_workers.rq_harness import RQClientQueue, RQJobLike, RQRetryLike

from art_trainer.core import _test_hooks


class RQSettings(NamedTuple):
    """RQ queue settings."""

    job_timeout_sec: int
    result_ttl_sec: int
    failure_ttl_sec: int
    retry_max: int
    retry_intervals: list[int]


class RQEnqueuer:
    """Adapter for enqueueing jobs to RQ.

    Handles job serialization and queue management.
    """

    _queue: RQClientQueue
    _retry: RQRetryLike
    _settings: RQSettings

    def __init__(self, redis_url: str, settings: RQSettings) -> None:
        """Initialize RQ enqueuer.

        Args:
            redis_url: Redis connection URL.
            settings: RQ settings.
        """
        self._settings = settings
        conn = _test_hooks.rq_connection_factory(redis_url)
        self._queue = _test_hooks.rq_queue_factory("art-trainer", conn)
        self._retry = _test_hooks.rq_retry_factory(
            max_retries=settings.retry_max,
            intervals=settings.retry_intervals,
        )

    def enqueue(
        self,
        func_path: str,
        payload: JSONObject,
        *,
        description: str,
    ) -> str:
        """Enqueue a job to the queue.

        Args:
            func_path: Dotted path to the job function.
            payload: Job payload.
            description: Human-readable job description.

        Returns:
            Job ID.
        """
        job: RQJobLike = self._queue.enqueue(
            func_path,
            payload,
            job_timeout=self._settings.job_timeout_sec,
            result_ttl=self._settings.result_ttl_sec,
            failure_ttl=self._settings.failure_ttl_sec,
            retry=self._retry,
            description=description,
        )
        return job.get_id()


__all__ = [
    "RQEnqueuer",
    "RQSettings",
]
