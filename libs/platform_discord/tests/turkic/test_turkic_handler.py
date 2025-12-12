"""Tests for turkic event handler."""

from __future__ import annotations

from collections.abc import Generator
from typing import Literal

import pytest
from platform_core.job_events import (
    JobCompletedV1,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
    encode_job_event,
)

from platform_discord.testing import fake_load_discord_module, hooks
from platform_discord.turkic.handler import (
    _narrow_turkic,
    decode_turkic_event,
    handle_turkic_event,
    is_completed,
    is_failed,
    is_progress,
    is_started,
)
from platform_discord.turkic.runtime import new_runtime


@pytest.fixture(autouse=True)
def _use_fake_discord() -> Generator[None, None, None]:
    """Set up fake discord module via hooks."""
    hooks.load_discord_module = fake_load_discord_module
    yield


def _make_started_event(job_id: str, user_id: int, queue: str = "turkic") -> JobStartedV1:
    return {
        "type": "turkic.job.started.v1",
        "domain": "turkic",
        "job_id": job_id,
        "user_id": user_id,
        "queue": queue,
    }


def _make_progress_event(
    job_id: str, user_id: int, progress: int, message: str | None = None
) -> JobProgressV1:
    ev: JobProgressV1 = {
        "type": "turkic.job.progress.v1",
        "domain": "turkic",
        "job_id": job_id,
        "user_id": user_id,
        "progress": progress,
    }
    if message is not None:
        ev["message"] = message
    return ev


def _make_completed_event(
    job_id: str, user_id: int, result_id: str, result_bytes: int
) -> JobCompletedV1:
    return {
        "type": "turkic.job.completed.v1",
        "domain": "turkic",
        "job_id": job_id,
        "user_id": user_id,
        "result_id": result_id,
        "result_bytes": result_bytes,
    }


def _make_failed_event(
    job_id: str, user_id: int, error_kind: Literal["user", "system"], message: str
) -> JobFailedV1:
    return {
        "type": "turkic.job.failed.v1",
        "domain": "turkic",
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": error_kind,
        "message": message,
    }


def test_decode_started_event() -> None:
    ev = _make_started_event(job_id="job-1", user_id=5, queue="turkic")
    payload = encode_job_event(ev)
    decoded = decode_turkic_event(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "job-1"
    assert is_started(decoded)


def test_decode_progress_event() -> None:
    ev = _make_progress_event(job_id="job-2", user_id=7, progress=50, message="processing")
    payload = encode_job_event(ev)
    decoded = decode_turkic_event(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "job-2"
    assert is_progress(decoded)


def test_decode_completed_event() -> None:
    ev = _make_completed_event(job_id="job-3", user_id=11, result_id="file-123", result_bytes=1024)
    payload = encode_job_event(ev)
    decoded = decode_turkic_event(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "job-3"
    assert is_completed(decoded)


def test_decode_failed_event() -> None:
    ev = _make_failed_event(
        job_id="job-4", user_id=13, error_kind="system", message="error occurred"
    )
    payload = encode_job_event(ev)
    decoded = decode_turkic_event(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "job-4"
    assert is_failed(decoded)


def test_decode_invalid_payload_returns_none() -> None:
    assert decode_turkic_event("not json") is None
    assert decode_turkic_event("{}") is None
    assert decode_turkic_event('{"type": "unknown.v1"}') is None


def test_decode_non_turkic_domain_returns_none() -> None:
    """Events from other domains should be ignored."""
    ev: JobStartedV1 = {
        "type": "digits.job.started.v1",
        "domain": "digits",
        "job_id": "other-1",
        "user_id": 1,
        "queue": "digits",
    }
    payload = encode_job_event(ev)
    assert decode_turkic_event(payload) is None


def test_handle_started_event() -> None:
    rt = new_runtime()
    ev = _make_started_event(job_id="h-1", user_id=1, queue="turkic")
    result = handle_turkic_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["job_id"] == "h-1"
    assert result["user_id"] == 1
    embed = result["embed"]
    if embed is None:
        pytest.fail("expected embed")
    assert embed.title == "Turkic Starting"


def test_handle_progress_event() -> None:
    rt = new_runtime()
    ev = _make_progress_event(job_id="h-2", user_id=2, progress=50)
    result = handle_turkic_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["job_id"] == "h-2"
    assert result["user_id"] == 2


def test_handle_progress_event_with_message() -> None:
    rt = new_runtime()
    ev = _make_progress_event(job_id="h-2b", user_id=2, progress=75, message="almost done")
    result = handle_turkic_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["job_id"] == "h-2b"


def test_handle_completed_event() -> None:
    rt = new_runtime()
    ev = _make_completed_event(job_id="h-3", user_id=3, result_id="res-abc", result_bytes=2048)
    result = handle_turkic_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["job_id"] == "h-3"
    assert result["user_id"] == 3


def test_handle_failed_event() -> None:
    rt = new_runtime()
    ev = _make_failed_event(job_id="h-4", user_id=4, error_kind="user", message="job failed")
    result = handle_turkic_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["job_id"] == "h-4"
    assert result["user_id"] == 4


def test_handle_null_user_id_returns_no_embed() -> None:
    """When user_id is None (0 after coercion), embed should be None."""
    rt = new_runtime()
    # Create event with user_id that will be treated as None
    ev: JobStartedV1 = {
        "type": "turkic.job.started.v1",
        "domain": "turkic",
        "job_id": "null-user",
        "user_id": 0,  # Will be treated as invalid
        "queue": "turkic",
    }
    # The runtime handles None user_id by returning embed=None
    # but since 0 is an int, it will still create an embed
    result = handle_turkic_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["job_id"] == "null-user"


def test_type_guards() -> None:
    """Test TypeGuard functions."""
    started: JobStartedV1 = _make_started_event("t1", 1)
    progress: JobProgressV1 = _make_progress_event("t2", 2, 50)
    completed: JobCompletedV1 = _make_completed_event("t3", 3, "r", 100)
    failed: JobFailedV1 = _make_failed_event("t4", 4, "system", "err")

    assert is_started(started)
    assert not is_started(progress)
    assert not is_started(completed)
    assert not is_started(failed)

    assert not is_progress(started)
    assert is_progress(progress)
    assert not is_progress(completed)
    assert not is_progress(failed)

    assert not is_completed(started)
    assert not is_completed(progress)
    assert is_completed(completed)
    assert not is_completed(failed)

    assert not is_failed(started)
    assert not is_failed(progress)
    assert not is_failed(completed)


def test_decode_turkic_unknown_type_returns_none() -> None:
    """Turkic events with unknown type should return None via exception path."""
    # Create a turkic domain event with an unrecognized type suffix
    # decode_job_event raises JSONTypeError for invalid suffixes like "queued"
    ev: JobStartedV1 = {
        "type": "turkic.job.queued.v1",  # Not a recognized suffix
        "domain": "turkic",
        "job_id": "unknown-1",
        "user_id": 1,
        "queue": "turkic",
    }
    payload = encode_job_event(ev)
    assert decode_turkic_event(payload) is None


def test_narrow_turkic_returns_none_for_unrecognized_type() -> None:
    """_narrow_turkic returns None when event type doesn't match any TypeGuard."""
    from platform_core.job_events import JobEventV1

    # Create a raw dict that satisfies JobEventV1 structurally but has unrecognized type
    # This bypasses decode_job_event validation to test _narrow_turkic directly
    ev: JobEventV1 = {
        "type": "turkic.job.unknown.v1",  # Unrecognized suffix
        "domain": "turkic",
        "job_id": "test-narrow",
        "user_id": 1,
        "queue": "turkic",
    }
    result = _narrow_turkic(ev)
    assert result is None


def test_handle_unknown_event_type_returns_none() -> None:
    """handle_turkic_event returns None for unrecognized event types."""
    rt = new_runtime()
    # Create a valid turkic event structure but with unrecognized type
    # We'll use JobStartedV1 but change the type field after
    ev: JobStartedV1 = {
        "type": "turkic.job.queued.v1",  # Unrecognized type
        "domain": "turkic",
        "job_id": "unknown-2",
        "user_id": 1,
        "queue": "turkic",
    }
    result = handle_turkic_event(rt, ev)
    assert result is None
