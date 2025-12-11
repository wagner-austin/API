"""Tests for transcript_api.events publish functions."""

from __future__ import annotations

from typing import TypeGuard

import pytest
from platform_core.config import _test_hooks as platform_hooks
from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    decode_job_event,
    default_events_channel,
)
from platform_core.json_utils import JSONTypeError
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
    tev.publish_failed(request_id="r2", user_id=9, error_kind="user", message="bad")
    tev.publish_failed(request_id="r3", user_id=11, error_kind="system", message="boom")

    assert stub.closed is True
    assert len(stub.published) == 3
    ch0, msg0 = stub.published[0]
    ch1, msg1 = stub.published[1]
    ch2, msg2 = stub.published[2]
    expected_channel = default_events_channel("transcript")
    assert ch0 == expected_channel and ch1 == expected_channel and ch2 == expected_channel
    ev0 = _require_event(msg0)
    ev1 = _require_event(msg1)
    ev2 = _require_event(msg2)
    assert _is_completed(ev0)
    assert ev0["domain"] == "transcript"
    assert ev0["job_id"] == "r1"
    assert ev0["user_id"] == 7
    assert ev0["result_id"] == "https://x"
    assert ev0["result_bytes"] == len(b"hello")

    assert _is_failed(ev1)
    assert ev1["error_kind"] == "user"
    assert _is_failed(ev2)
    assert ev2["error_kind"] == "system"
    stub.assert_only_called({"publish", "close"})


def test_publish_failed_rejects_invalid_kind() -> None:
    """Test _ensure_error_kind rejects invalid error kinds."""
    stub = FakeRedis()

    def _redis_loader(url: str) -> RedisStrProto:
        return stub

    platform_hooks.get_env = make_fake_env({"REDIS_URL": "redis://unit"})
    _test_hooks.redis_factory = _redis_loader

    with pytest.raises(JSONTypeError):
        _ = tev._ensure_error_kind("other")
    stub.assert_only_called(set())


def _require_event(payload: str) -> JobEventV1:
    ev = decode_job_event(payload)
    assert ev is not None
    assert ev["domain"] == "transcript"
    return ev


def _is_completed(ev: JobEventV1) -> TypeGuard[JobCompletedV1]:
    return ev["type"] == "transcript.job.completed.v1"


def _is_failed(ev: JobEventV1) -> TypeGuard[JobFailedV1]:
    return ev["type"] == "transcript.job.failed.v1"
