from __future__ import annotations

from typing import TypeGuard

import pytest

from platform_core.job_events import (
    JobEventV1,
    JobFailedV1,
    decode_job_event,
    default_events_channel,
    encode_job_event,
    is_completed,
    is_failed,
    is_progress,
    is_started,
    make_completed_event,
    make_event_type,
    make_failed_event,
    make_progress_event,
    make_started_event,
)


def _roundtrip(event: JobEventV1) -> JobEventV1:
    encoded = encode_job_event(event)
    return decode_job_event(encoded)


def _is_failed(event: JobEventV1) -> TypeGuard[JobFailedV1]:
    return "error_kind" in event


def test_event_type_and_channel() -> None:
    assert make_event_type("turkic", "started") == "turkic.job.started.v1"
    assert default_events_channel("transcript") == "transcript:events"
    assert default_events_channel("qr") == "qr:events"
    assert default_events_channel("music_wrapped") == "music_wrapped:events"


def test_roundtrip_started_event() -> None:
    event = make_started_event(domain="turkic", job_id="j1", user_id=5, queue="primary")
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_started_event_qr_domain() -> None:
    event = make_started_event(domain="qr", job_id="q1", user_id=9, queue="qr-queue")
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_progress_event_with_message() -> None:
    event = make_progress_event(
        domain="digits",
        job_id="j2",
        user_id=7,
        progress=42,
        message="working",
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_progress_event_with_payload() -> None:
    event = make_progress_event(
        domain="digits",
        job_id="j2",
        user_id=7,
        progress=42,
        message="working",
        payload={"metrics": {"loss": 0.1}},
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_progress_event_without_message() -> None:
    event = make_progress_event(domain="trainer", job_id="j3", user_id=1, progress=99, message=None)
    decoded = _roundtrip(event)
    assert decoded == {
        "type": "trainer.job.progress.v1",
        "domain": "trainer",
        "job_id": "j3",
        "user_id": 1,
        "progress": 99,
    }


def test_roundtrip_completed_event() -> None:
    event = make_completed_event(
        domain="databank",
        job_id="j4",
        user_id=11,
        result_id="file-1",
        result_bytes=2048,
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_failed_event_user_kind() -> None:
    event = make_failed_event(
        domain="transcript",
        job_id="j5",
        user_id=13,
        error_kind="user",
        message="invalid input",
    )
    decoded = _roundtrip(event)
    assert decoded == event


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ('{"type": 1}', "job event type must be a string"),
        ('{"type": "x.y.z"}', "invalid job event type format"),
        ('{"type": "unknown.job.started.v1"}', "invalid domain in job event"),
        ('{"type": "turkic.job.invalid.v1"}', "invalid event suffix in job event"),
        (
            '{"type": "turkic.job.started.v1", "domain": 1}',
            "job event domain must be a string",
        ),
        (
            '{"type": "turkic.job.started.v1", "domain": "transcript"}',
            "job event domain mismatch",
        ),
        (
            '{"type": "turkic.job.started.v1", "domain": "turkic", "job_id": 1, "user_id": 2}',
            "job_id and user_id are required in job event",
        ),
        (
            '{"type": "turkic.job.started.v1", "domain": "turkic", "job_id": "j", "user_id": 1}',
            "queue is required in started event",
        ),
        (
            '{"type": "turkic.job.progress.v1", "domain": "turkic", "job_id": "j", "user_id": 1}',
            "progress must be an int in progress event",
        ),
        (
            '{"type": "turkic.job.completed.v1", "domain": "turkic", "job_id": "j", "user_id": 1}',
            "completed event requires result_id and result_bytes",
        ),
        (
            '{"type": "turkic.job.failed.v1", "domain": "turkic", '
            '"job_id": "j", "user_id": 1, "message": "x"}',
            "failed event requires error_kind and message",
        ),
        (
            '{"type": "turkic.job.failed.v1", "domain": "turkic", '
            '"job_id": "j", "user_id": 1, "error_kind": "oops", "message": "x"}',
            "invalid error_kind in failed event",
        ),
    ],
)
def test_decode_job_event_invalid(payload: str, expected_message: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        decode_job_event(payload)
    assert expected_message in str(excinfo.value)


def test_decode_job_event_requires_object() -> None:
    with pytest.raises(ValueError):
        decode_job_event("[]")


def test_failed_event_system_kind() -> None:
    payload = encode_job_event(
        make_failed_event(
            domain="turkic",
            job_id="j6",
            user_id=22,
            error_kind="system",
            message="boom",
        )
    )
    decoded = decode_job_event(payload)
    assert _is_failed(decoded)
    assert decoded["error_kind"] == "system"


def test_is_started_typeguard() -> None:
    started = make_started_event(domain="turkic", job_id="j1", user_id=1, queue="q")
    progress = make_progress_event(domain="turkic", job_id="j2", user_id=1, progress=50)
    completed = make_completed_event(
        domain="turkic", job_id="j3", user_id=1, result_id="r", result_bytes=100
    )
    failed = make_failed_event(
        domain="turkic", job_id="j4", user_id=1, error_kind="system", message="err"
    )
    assert is_started(started) is True
    assert is_started(progress) is False
    assert is_started(completed) is False
    assert is_started(failed) is False


def test_is_progress_typeguard() -> None:
    started = make_started_event(domain="turkic", job_id="j1", user_id=1, queue="q")
    progress = make_progress_event(domain="turkic", job_id="j2", user_id=1, progress=50)
    completed = make_completed_event(
        domain="turkic", job_id="j3", user_id=1, result_id="r", result_bytes=100
    )
    failed = make_failed_event(
        domain="turkic", job_id="j4", user_id=1, error_kind="system", message="err"
    )
    assert is_progress(started) is False
    assert is_progress(progress) is True
    assert is_progress(completed) is False
    assert is_progress(failed) is False


def test_is_completed_typeguard() -> None:
    started = make_started_event(domain="turkic", job_id="j1", user_id=1, queue="q")
    progress = make_progress_event(domain="turkic", job_id="j2", user_id=1, progress=50)
    completed = make_completed_event(
        domain="turkic", job_id="j3", user_id=1, result_id="r", result_bytes=100
    )
    failed = make_failed_event(
        domain="turkic", job_id="j4", user_id=1, error_kind="system", message="err"
    )
    assert is_completed(started) is False
    assert is_completed(progress) is False
    assert is_completed(completed) is True
    assert is_completed(failed) is False


def test_is_failed_typeguard() -> None:
    started = make_started_event(domain="turkic", job_id="j1", user_id=1, queue="q")
    progress = make_progress_event(domain="turkic", job_id="j2", user_id=1, progress=50)
    completed = make_completed_event(
        domain="turkic", job_id="j3", user_id=1, result_id="r", result_bytes=100
    )
    failed = make_failed_event(
        domain="turkic", job_id="j4", user_id=1, error_kind="system", message="err"
    )
    assert is_failed(started) is False
    assert is_failed(progress) is False
    assert is_failed(completed) is False
    assert is_failed(failed) is True
