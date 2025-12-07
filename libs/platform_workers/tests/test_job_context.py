from __future__ import annotations

from typing import TypeGuard

from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    JobProgressV1,
    decode_job_event,
)

from platform_workers.job_context import JobContext, make_job_context
from platform_workers.testing import FakeRedis


def _last(redis: FakeRedis) -> JobEventV1:
    assert redis.published
    pub = redis.published[-1]
    return decode_job_event(pub.payload)


def _is_progress(event: JobEventV1) -> TypeGuard[JobProgressV1]:
    return "progress" in event


def _is_completed(event: JobEventV1) -> TypeGuard[JobCompletedV1]:
    return "result_id" in event


def _is_failed(event: JobEventV1) -> TypeGuard[JobFailedV1]:
    return "error_kind" in event


def test_job_context_publishes_started_and_progress() -> None:
    redis = FakeRedis()
    ctx: JobContext = make_job_context(
        redis=redis,
        domain="turkic",
        events_channel="turkic:events",
        job_id="job-1",
        user_id=7,
        queue_name="queue-x",
    )

    ctx.publish_started()
    started = _last(redis)
    assert started == {
        "type": "turkic.job.started.v1",
        "domain": "turkic",
        "job_id": "job-1",
        "user_id": 7,
        "queue": "queue-x",
    }

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
        domain="transcript",
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

    ctx.publish_failed("system", "boom")
    failed = _last(redis)
    assert _is_failed(failed)
    assert failed["error_kind"] == "system"
    assert failed["message"] == "boom"

    ctx.publish_failed("user", "bad input")
    failed_user = _last(redis)
    assert _is_failed(failed_user)
    assert failed_user["error_kind"] == "user"

    redis.assert_only_called({"publish"})
