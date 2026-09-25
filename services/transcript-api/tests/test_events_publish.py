"""Tests for transcript_api.events publish functions."""

from __future__ import annotations

from platform_core.config import _test_hooks as platform_hooks
from platform_core.job_events import (
    ErrorKind,
    JobCompletedV1,
    JobDomain,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
    decode_job_event,
    default_events_channel,
    is_completed,
    is_failed,
)
from platform_core.testing import make_fake_env
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from transcript_api import _test_hooks
from transcript_api import events as tev


def test_publish_completed_and_failed() -> None:
    """Test publish_completed and publish_failed send correct events."""
    stub = FakeRedis()

    def _redis_loader(url: str) -> RedisStrProto:
        return stub

    platform_hooks.get_env = make_fake_env({"REDIS_URL": "redis://unit"})
    _test_hooks.redis_factory = _redis_loader

    tev.publish_completed(request_id="r1", user_id=7, url="https://x", text="hello")
    tev.publish_failed(request_id="r2", user_id=9, error_kind=ErrorKind.USER, message="bad")
    tev.publish_failed(request_id="r3", user_id=11, error_kind=ErrorKind.SYSTEM, message="boom")

    assert stub.closed is True
    assert len(stub.published) == 3
    ch0, msg0 = stub.published[0]
    ch1, msg1 = stub.published[1]
    ch2, msg2 = stub.published[2]
    expected_channel = default_events_channel(JobDomain.TRANSCRIPT)
    assert ch0 == expected_channel and ch1 == expected_channel and ch2 == expected_channel
    ev0 = _require_event(msg0)
    ev1 = _require_event(msg1)
    ev2 = _require_event(msg2)
    assert is_completed(ev0)
    assert ev0["type"] == "transcript.job.completed.v1"
    assert ev0["job_id"] == "r1"
    assert ev0["user_id"] == 7
    assert ev0["result_id"] == "https://x"
    assert ev0["result_bytes"] == len(b"hello")

    assert is_failed(ev1)
    assert ev1["type"] == "transcript.job.failed.v1"
    assert ev1["error_kind"] is ErrorKind.USER
    assert ev1["message"] == "bad"
    assert is_failed(ev2)
    assert ev2["error_kind"] is ErrorKind.SYSTEM
    assert ev2["message"] == "boom"
    stub.assert_only_called({"publish", "close"})


def _require_event(payload: str) -> JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1:
    ev = decode_job_event(payload)
    assert ev["domain"] is JobDomain.TRANSCRIPT
    return ev
