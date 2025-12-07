from __future__ import annotations

from typing import TypeGuard

from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
    decode_job_event,
    default_events_channel,
)
from platform_core.queues import TURKIC_QUEUE
from platform_workers.job_context import make_job_context
from platform_workers.testing import FakeRedis


def _is_started(ev: JobEventV1) -> TypeGuard[JobStartedV1]:
    return ev["type"] == "turkic.job.started.v1"


def _is_progress(ev: JobEventV1) -> TypeGuard[JobProgressV1]:
    return ev["type"] == "turkic.job.progress.v1"


def _is_completed(ev: JobEventV1) -> TypeGuard[JobCompletedV1]:
    return ev["type"] == "turkic.job.completed.v1"


def _is_failed(ev: JobEventV1) -> TypeGuard[JobFailedV1]:
    return ev["type"] == "turkic.job.failed.v1"


def test_job_context_progress_message_optional() -> None:
    redis = FakeRedis()
    ctx = make_job_context(
        redis=redis,
        domain="turkic",
        events_channel=default_events_channel("turkic"),
        job_id="job1",
        user_id=42,
        queue_name=TURKIC_QUEUE,
    )
    ctx.publish_progress(10, None)
    ctx.publish_progress(20, "processing")

    decoded = [decode_job_event(payload) for _, payload in redis.published]
    first = decoded[0]
    second = decoded[1]
    assert _is_progress(first)
    assert "message" not in first
    assert _is_progress(second)
    assert second["message"] == "processing"
    redis.assert_only_called({"publish"})


def test_job_context_started_and_completed_and_failed() -> None:
    redis = FakeRedis()
    ctx = make_job_context(
        redis=redis,
        domain="turkic",
        events_channel=default_events_channel("turkic"),
        job_id="job2",
        user_id=42,
        queue_name=TURKIC_QUEUE,
    )
    ctx.publish_started()
    ctx.publish_completed("fid", 1024)
    ctx.publish_failed("user", "msg")

    decoded: list[JobEventV1] = [decode_job_event(payload) for _, payload in redis.published]
    started = decoded[0]
    completed = decoded[1]
    failed = decoded[2]

    assert _is_started(started)
    assert started["queue"] == TURKIC_QUEUE
    assert _is_completed(completed)
    assert completed["result_id"] == "fid"
    assert completed["result_bytes"] == 1024
    assert _is_failed(failed)
    assert failed["error_kind"] == "user"
    assert failed["message"] == "msg"
    redis.assert_only_called({"publish"})
