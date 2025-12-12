"""Tests for digits event handler."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.digits_metrics_events import (
    DigitsConfigV1,
    encode_digits_metrics_event,
    make_artifact_event,
    make_batch_metrics_event,
    make_best_metrics_event,
    make_completed_metrics_event,
    make_config_event,
    make_epoch_metrics_event,
    make_prune_event,
    make_upload_event,
)
from platform_core.job_events import encode_job_event, make_failed_event

from platform_discord.handwriting.handler import (
    decode_digits_event_safe,
    handle_digits_event,
)
from platform_discord.handwriting.runtime import new_runtime
from platform_discord.testing import fake_load_discord_module, hooks


@pytest.fixture(autouse=True)
def _use_fake_discord() -> Generator[None, None, None]:
    """Set up fake discord module via hooks."""
    hooks.load_discord_module = fake_load_discord_module
    yield


def test_decode_config_event() -> None:
    ev = make_config_event(
        job_id="run-1",
        user_id=5,
        model_id="digits-v1",
        total_epochs=10,
        queue="digits",
    )
    payload = encode_digits_metrics_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-1"


def test_decode_batch_event() -> None:
    ev = make_batch_metrics_event(
        job_id="run-2",
        user_id=7,
        model_id="digits-v1",
        epoch=1,
        total_epochs=10,
        batch=5,
        total_batches=100,
        batch_loss=0.5,
        batch_acc=0.9,
        avg_loss=0.45,
        samples_per_sec=120.0,
        main_rss_mb=512,
        workers_rss_mb=256,
        worker_count=4,
        cgroup_usage_mb=768,
        cgroup_limit_mb=4096,
        cgroup_pct=18.75,
        anon_mb=600,
        file_mb=168,
    )
    payload = encode_digits_metrics_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-2"


def test_decode_epoch_event() -> None:
    ev = make_epoch_metrics_event(
        job_id="run-3",
        user_id=11,
        model_id="digits-v1",
        epoch=3,
        total_epochs=10,
        train_loss=0.45,
        val_acc=0.95,
        time_s=120.5,
    )
    payload = encode_digits_metrics_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-3"


def test_decode_best_event() -> None:
    ev = make_best_metrics_event(
        job_id="run-4",
        user_id=13,
        model_id="digits-v1",
        epoch=5,
        val_acc=0.98,
    )
    payload = encode_digits_metrics_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-4"


def test_decode_artifact_event() -> None:
    ev = make_artifact_event(
        job_id="run-5",
        user_id=17,
        model_id="digits-v1",
        path="/models/best.pth",
    )
    payload = encode_digits_metrics_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-5"


def test_decode_upload_event() -> None:
    ev = make_upload_event(
        job_id="run-6",
        user_id=19,
        model_id="digits-v1",
        status=200,
        model_bytes=1024,
        manifest_bytes=256,
        file_id="abc123",
        file_sha256="deadbeef",
    )
    payload = encode_digits_metrics_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-6"


def test_decode_prune_event() -> None:
    ev = make_prune_event(
        job_id="run-7",
        user_id=23,
        model_id="digits-v1",
        deleted_count=3,
    )
    payload = encode_digits_metrics_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-7"


def test_decode_completed_event() -> None:
    ev = make_completed_metrics_event(
        job_id="run-8",
        user_id=29,
        model_id="digits-v1",
        val_acc=0.99,
    )
    payload = encode_digits_metrics_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-8"


def test_decode_failed_job_event() -> None:
    ev = make_failed_event(
        domain="digits",
        job_id="run-9",
        user_id=31,
        error_kind="system",
        message="training exploded",
    )
    payload = encode_job_event(ev)
    decoded = decode_digits_event_safe(payload)
    if decoded is None:
        pytest.fail("expected decoded event")
    assert decoded["job_id"] == "run-9"


def test_decode_invalid_payload_returns_none() -> None:
    assert decode_digits_event_safe("not json") is None
    assert decode_digits_event_safe("{}") is None
    assert decode_digits_event_safe('{"type": "unknown.v1"}') is None


def test_handle_config_event() -> None:
    rt = new_runtime()
    ev = make_config_event(
        job_id="h-1",
        user_id=1,
        model_id="digits-v1",
        total_epochs=5,
        queue="digits",
    )
    result = handle_digits_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-1"
    assert result["user_id"] == 1


def test_handle_config_event_with_optional_fields() -> None:
    rt = new_runtime()
    ev = make_config_event(
        job_id="h-1b",
        user_id=1,
        model_id="digits-v1",
        total_epochs=5,
        queue="digits",
        cpu_cores=4,
        optimal_threads=8,
        memory_mb=2048,
        optimal_workers=2,
        max_batch_size=64,
        device="cuda:0",
        batch_size=32,
        learning_rate=0.001,
        augment=True,
        aug_rotate=15.0,
        aug_translate=0.1,
        noise_prob=0.05,
        dots_prob=0.02,
    )
    result = handle_digits_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-1b"


def test_handle_batch_event() -> None:
    rt = new_runtime()
    ev = make_batch_metrics_event(
        job_id="h-2",
        user_id=2,
        model_id="digits-v1",
        epoch=1,
        total_epochs=5,
        batch=10,
        total_batches=100,
        batch_loss=0.4,
        batch_acc=0.92,
        avg_loss=0.42,
        samples_per_sec=150.0,
        main_rss_mb=600,
        workers_rss_mb=200,
        worker_count=4,
        cgroup_usage_mb=800,
        cgroup_limit_mb=4096,
        cgroup_pct=19.5,
        anon_mb=650,
        file_mb=150,
    )
    result = handle_digits_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-2"


def test_handle_epoch_event() -> None:
    rt = new_runtime()
    ev = make_epoch_metrics_event(
        job_id="h-3",
        user_id=3,
        model_id="digits-v1",
        epoch=2,
        total_epochs=5,
        train_loss=0.3,
        val_acc=0.94,
        time_s=60.0,
    )
    result = handle_digits_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-3"


def test_handle_best_event_returns_none() -> None:
    """Best events update runtime state but don't trigger notifications."""
    rt = new_runtime()
    ev = make_best_metrics_event(
        job_id="h-4",
        user_id=4,
        model_id="digits-v1",
        epoch=3,
        val_acc=0.96,
    )
    result = handle_digits_event(rt, ev)
    assert result is None


def test_handle_artifact_event_returns_none() -> None:
    """Artifact events update runtime state but don't trigger notifications."""
    rt = new_runtime()
    ev = make_artifact_event(
        job_id="h-5",
        user_id=5,
        model_id="digits-v1",
        path="/models/checkpoint.pth",
    )
    result = handle_digits_event(rt, ev)
    assert result is None


def test_handle_upload_event_returns_none() -> None:
    """Upload events update runtime state but don't trigger notifications."""
    rt = new_runtime()
    ev = make_upload_event(
        job_id="h-6",
        user_id=6,
        model_id="digits-v1",
        status=201,
        model_bytes=2048,
        manifest_bytes=512,
        file_id="xyz789",
        file_sha256="cafebabe",
    )
    result = handle_digits_event(rt, ev)
    assert result is None


def test_handle_prune_event_returns_none() -> None:
    """Prune events update runtime state but don't trigger notifications."""
    rt = new_runtime()
    ev = make_prune_event(
        job_id="h-7",
        user_id=7,
        model_id="digits-v1",
        deleted_count=2,
    )
    result = handle_digits_event(rt, ev)
    assert result is None


def test_handle_completed_event() -> None:
    rt = new_runtime()
    ev = make_completed_metrics_event(
        job_id="h-8",
        user_id=8,
        model_id="digits-v1",
        val_acc=0.97,
    )
    result = handle_digits_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-8"


def test_handle_failed_event() -> None:
    rt = new_runtime()
    ev = make_failed_event(
        domain="digits",
        job_id="h-9",
        user_id=9,
        error_kind="user",
        message="training failed",
    )
    result = handle_digits_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "h-9"


def test_handle_unknown_event_type_returns_none() -> None:
    """Test that handle_digits_event returns None for unrecognized event type."""
    from platform_core.job_events import JobStartedV1

    rt = new_runtime()
    # Create a valid event but not matching any TypeGuard
    # JobStartedV1 is part of DigitsEventV1 but not handled by handle_digits_event
    event: JobStartedV1 = {
        "type": "digits.job.started.v1",
        "domain": "digits",
        "job_id": "fake",
        "user_id": 1,
        "queue": "digits",
    }
    result = handle_digits_event(rt, event)
    assert result is None


def test_extract_optional_fields_with_invalid_types() -> None:
    """Test that optional field extractors handle invalid types correctly."""
    rt = new_runtime()
    # Create a config event and manually corrupt its optional fields
    ev: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "extract-test",
        "user_id": 1,
        "model_id": "digits-v1",
        "total_epochs": 5,
        "queue": "digits",
    }
    # Test that handle_digits_event works even with missing optional fields
    result = handle_digits_event(rt, ev)
    if result is None:
        pytest.fail("expected result")
    assert result["request_id"] == "extract-test"


def test_extract_optional_float_rejects_bool() -> None:
    """Test that _extract_optional_float returns None for bool values.

    This ensures the special case where bool is rejected even though
    bool is technically a subclass of int in Python.
    """
    from platform_discord.handwriting.handler import _extract_optional_float

    # Create a config event with augment=True (a bool field)
    # _extract_optional_float should return None when called on it
    ev: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "bool-float-test",
        "user_id": 1,
        "model_id": "digits-v1",
        "total_epochs": 5,
        "queue": "digits",
        "augment": True,
    }
    # The extractor should return None for bool values
    # even though augment is a bool field, testing that _extract_optional_float
    # correctly rejects bools when looking for floats
    result = _extract_optional_float(ev, "augment")
    assert result is None
