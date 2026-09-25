from __future__ import annotations

from typing import TypeGuard

from platform_core.job_events import (
    ErrorKind,
    JobCompletedV1,
    JobDomain,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
    decode_job_event,
)

from platform_workers.job_context import JobContext, make_job_context
from platform_workers.testing import FakeRedis


def _last(redis: FakeRedis) -> JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1:
    assert redis.published
    pub = redis.published[-1]
    return decode_job_event(pub.payload)


def _is_progress(
    event: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> TypeGuard[JobProgressV1]:
    return "progress" in event


def _is_completed(
    event: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> TypeGuard[JobCompletedV1]:
    return "result_id" in event


def _is_failed(
    event: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> TypeGuard[JobFailedV1]:
    return "error_kind" in event


def test_job_context_publishes_started_and_progress() -> None:
    redis = FakeRedis()
    ctx: JobContext = make_job_context(
        redis=redis,
        domain=JobDomain.TURKIC,
        events_channel="turkic:events",
        job_id="job-1",
        user_id=7,
        queue_name="queue-x",
    )

    ctx.publish_started()
    started = _last(redis)
    assert started == {
        "type": "turkic.job.started.v1",
        "domain": JobDomain.TURKIC,
        "job_id": "job-1",
        "user_id": 7,
        "queue": "queue-x",
    }
    assert '"domain":"turkic"' in redis.published[-1].payload.replace(" ", "")

    ctx.publish_progress(50, "halfway")
    progress = _last(redis)
    assert _is_progress(progress)
    assert progress["progress"] == 50
    assert progress["message"] == "halfway"

    ctx.publish_progress(75, None)
    progress_no_msg = _last(redis)
    assert _is_progress(progress_no_msg)
    assert "message" not in progress_no_msg

    redis.assert_only_called({"publish"})


def test_job_context_publishes_completion_and_failure() -> None:
    redis = FakeRedis()
    ctx: JobContext = make_job_context(
        redis=redis,
        domain=JobDomain.TRANSCRIPT,
        events_channel="transcript:events",
        job_id="job-2",
        user_id=11,
        queue_name="queue-y",
    )

    ctx.publish_completed("result-1", 4096)
    completed = _last(redis)
    assert _is_completed(completed)
    assert completed["result_id"] == "result-1"
    assert completed["result_bytes"] == 4096

    ctx.publish_failed(ErrorKind.SYSTEM, "boom")
    failed = _last(redis)
    assert _is_failed(failed)
    assert failed["error_kind"] is ErrorKind.SYSTEM
    assert failed["message"] == "boom"

    ctx.publish_failed(ErrorKind.USER, "bad input")
    failed_user = _last(redis)
    assert _is_failed(failed_user)
    assert failed_user["error_kind"] is ErrorKind.USER

    redis.assert_only_called({"publish"})
