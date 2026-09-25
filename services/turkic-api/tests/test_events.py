from __future__ import annotations

from platform_core.job_events import (
    ErrorKind,
    JobDomain,
    decode_job_event,
    default_events_channel,
    is_completed,
    is_failed,
    is_progress,
    is_started,
)
from platform_core.queues import TURKIC_QUEUE
from platform_workers.job_context import make_job_context
from platform_workers.testing import FakeRedis


def test_job_context_progress_message_optional() -> None:
    redis = FakeRedis()
    ctx = make_job_context(
        redis=redis,
        domain=JobDomain.TURKIC,
        events_channel=default_events_channel(JobDomain.TURKIC),
        job_id="job1",
        user_id=42,
        queue_name=TURKIC_QUEUE,
    )
    ctx.publish_progress(10, None)
    ctx.publish_progress(20, "processing")

    decoded = [decode_job_event(payload) for _, payload in redis.published]
    first = decoded[0]
    second = decoded[1]
    assert is_progress(first)
    assert first["type"] == "turkic.job.progress.v1"
    assert "message" not in first
    assert is_progress(second)
    assert second["message"] == "processing"
    redis.assert_only_called({"publish"})


def test_job_context_started_and_completed_and_failed() -> None:
    redis = FakeRedis()
    ctx = make_job_context(
        redis=redis,
        domain=JobDomain.TURKIC,
        events_channel=default_events_channel(JobDomain.TURKIC),
        job_id="job2",
        user_id=42,
        queue_name=TURKIC_QUEUE,
    )
    ctx.publish_started()
    ctx.publish_completed("fid", 1024)
    ctx.publish_failed(ErrorKind.USER, "msg")

    decoded = [decode_job_event(payload) for _, payload in redis.published]
    started = decoded[0]
    completed = decoded[1]
    failed = decoded[2]

    assert is_started(started)
    assert started["type"] == "turkic.job.started.v1"
    assert started["queue"] == TURKIC_QUEUE
    assert is_completed(completed)
    assert completed["type"] == "turkic.job.completed.v1"
    assert completed["result_id"] == "fid"
    assert completed["result_bytes"] == 1024
    assert is_failed(failed)
    assert failed["type"] == "turkic.job.failed.v1"
    assert failed["error_kind"] is ErrorKind.USER
    assert failed["message"] == "msg"
    redis.assert_only_called({"publish"})
