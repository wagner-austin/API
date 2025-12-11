"""Tests for trainer event handler."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.job_events import encode_job_event, make_failed_event
from platform_core.trainer_metrics_events import (
    encode_trainer_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_progress_metrics_event,
)

from platform_discord.testing import fake_load_discord_module, hooks
from platform_discord.trainer.handler import (
    decode_trainer_event,
    handle_trainer_event,
)
from platform_discord.trainer.runtime import new_runtime


@pytest.fixture(autouse=True)
def _use_fake_discord() -> Generator[None, None, None]:
    """Set up fake discord module via hooks."""
    hooks.load_discord_module = fake_load_discord_module
    yield


def test_decode_config_event() -> None:
    ev = make_config_event(
        job_id="run-1",
        user_id=5,
        model_family="gpt2",
        model_size="small",
        total_epochs=10,
        queue="primary",
    )
    payload = encode_trainer_metrics_event(ev)
    decoded = decode_trainer_event(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-1"


def test_decode_progress_event() -> None:
    ev = make_progress_metrics_event(
        job_id="run-2",
        user_id=7,
        epoch=3,
        total_epochs=10,
        step=150,
        train_loss=1.234,
        train_ppl=3.44,
        grad_norm=0.5,
        samples_per_sec=120.0,
    )
    payload = encode_trainer_metrics_event(ev)
    decoded = decode_trainer_event(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-2"


def test_decode_completed_event() -> None:
    ev = make_completed_metrics_event(
        job_id="run-3",
        user_id=11,
        test_loss=0.456,
        test_ppl=1.578,
        artifact_path="/path/to/model",
    )
    payload = encode_trainer_metrics_event(ev)
    decoded = decode_trainer_event(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-3"


def test_decode_failed_job_event() -> None:
    ev = make_failed_event(
        domain="trainer",
        job_id="run-4",
        user_id=13,
        error_kind="system",
        message="training exploded",
    )
    payload = encode_job_event(ev)
    decoded = decode_trainer_event(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-4"


def test_decode_invalid_payload_returns_none() -> None:
    assert decode_trainer_event("not json") is None
    assert decode_trainer_event("{}") is None
    assert decode_trainer_event('{"type": "unknown.v1"}') is None


def test_decode_non_failed_job_event_returns_none() -> None:
    """Test that non-failed job events (e.g., started, progress) return None."""
    from platform_core.job_events import encode_job_event, make_started_event

    # A started event for trainer domain should not be handled (only failed events)
    ev = make_started_event(domain="trainer", job_id="s-1", user_id=1, queue="q")
    payload = encode_job_event(ev)
    assert decode_trainer_event(payload) is None


def test_decode_non_trainer_failed_event_returns_none() -> None:
    """Test that failed job events for non-trainer domains return None."""
    ev = make_failed_event(
        domain="turkic",  # Not trainer domain
        job_id="t-1",
        user_id=1,
        error_kind="system",
        message="boom",
    )
    payload = encode_job_event(ev)
    assert decode_trainer_event(payload) is None


def test_handle_config_event() -> None:
    rt = new_runtime()
    ev = make_config_event(
        job_id="h-1",
        user_id=1,
        model_family="gpt2",
        model_size="small",
        total_epochs=5,
        queue="q",
    )
    result = handle_trainer_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-1"
    assert result["user_id"] == 1


def test_handle_progress_event() -> None:
    rt = new_runtime()
    ev = make_progress_metrics_event(
        job_id="h-2",
        user_id=2,
        epoch=1,
        total_epochs=5,
        step=10,
        train_loss=2.0,
        train_ppl=7.39,
        grad_norm=0.3,
        samples_per_sec=95.0,
    )
    result = handle_trainer_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-2"


def test_handle_completed_event() -> None:
    rt = new_runtime()
    ev = make_completed_metrics_event(
        job_id="h-3",
        user_id=3,
        test_loss=0.5,
        test_ppl=1.2,
        artifact_path="/out",
    )
    result = handle_trainer_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-3"


def test_handle_failed_event() -> None:
    rt = new_runtime()
    ev = make_failed_event(
        domain="trainer",
        job_id="h-4",
        user_id=4,
        error_kind="system",
        message="boom",
    )
    result = handle_trainer_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-4"


def test_handle_unknown_event_type_returns_none() -> None:
    """Test that handle_trainer_event returns None for unrecognized event type."""
    from platform_core.job_events import JobFailedV1

    rt = new_runtime()
    # Create a valid JobFailedV1 but with type that doesn't match is_job_failed TypeGuard
    # The TypeGuard checks for ".job.failed." in type, so we use a different pattern
    event: JobFailedV1 = {
        "type": "trainer.unexpected.event.v1",  # No ".job.failed." pattern
        "domain": "trainer",
        "job_id": "fake",
        "user_id": 1,
        "error_kind": "system",
        "message": "test",
    }
    result = handle_trainer_event(rt, event)
    assert result is None
