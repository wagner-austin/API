from __future__ import annotations

from typing import Protocol

from platform_core.job_events import (
    ErrorKind,
    JobDomain,
    JobEventV1,
    encode_job_event,
    make_completed_event,
    make_failed_event,
    make_progress_event,
    make_started_event,
)
from platform_core.json_utils import JSONValue

from platform_workers.redis import RedisStrProto


class JobContext(Protocol):
    """Protocol for publishing lifecycle events for a job."""

    def publish_started(self) -> None: ...

    def publish_progress(
        self, progress: int, message: str | None = None, *, payload: JSONValue | None = None
    ) -> None: ...

    def publish_completed(self, result_id: str, result_bytes: int) -> None: ...

    def publish_failed(self, error_kind: ErrorKind, message: str) -> None: ...


def make_job_context(
    *,
    redis: RedisStrProto,
    domain: JobDomain,
    events_channel: str,
    job_id: str,
    user_id: int,
    queue_name: str,
) -> JobContext:
    """Create a context for publishing job lifecycle events."""

    class _Ctx:
        def publish_started(self) -> None:
            event: JobEventV1 = make_started_event(
                domain=domain,
                job_id=job_id,
                user_id=user_id,
                queue=queue_name,
            )
            redis.publish(events_channel, encode_job_event(event))

        def publish_progress(
            self, progress: int, message: str | None = None, *, payload: JSONValue | None = None
        ) -> None:
            event: JobEventV1 = make_progress_event(
                domain=domain,
                job_id=job_id,
                user_id=user_id,
                progress=progress,
                message=message,
                payload=payload,
            )
            redis.publish(events_channel, encode_job_event(event))

        def publish_completed(self, result_id: str, result_bytes: int) -> None:
            event: JobEventV1 = make_completed_event(
                domain=domain,
                job_id=job_id,
                user_id=user_id,
                result_id=result_id,
                result_bytes=result_bytes,
            )
            redis.publish(events_channel, encode_job_event(event))

        def publish_failed(self, error_kind: ErrorKind, message: str) -> None:
            event: JobEventV1 = make_failed_event(
                domain=domain,
                job_id=job_id,
                user_id=user_id,
                error_kind=error_kind,
                message=message,
            )
            redis.publish(events_channel, encode_job_event(event))

    return _Ctx()


__all__ = ["JobContext", "make_job_context"]
