from __future__ import annotations

from typing import TypeGuard

import pytest

from platform_core.job_events import (
    ErrorKind,
    EventSuffix,
    JobCompletedV1,
    JobDomain,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
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
from platform_core.json_utils import JSONTypeError


def _roundtrip(
    event: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1:
    encoded = encode_job_event(event)
    return decode_job_event(encoded)


def _is_failed(
    event: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> TypeGuard[JobFailedV1]:
    return "error_kind" in event


def test_event_type_and_channel() -> None:
    assert make_event_type(JobDomain.TURKIC, EventSuffix.STARTED) == "turkic.job.started.v1"
    assert default_events_channel(JobDomain.TRANSCRIPT) == "transcript:events"
    assert default_events_channel(JobDomain.QR) == "qr:events"
    assert default_events_channel(JobDomain.MUSIC_WRAPPED) == "music_wrapped:events"


def test_vocabularies_carry_the_wire_words_in_order() -> None:
    assert [member.value for member in JobDomain] == [
        "covenant",
        "databank",
        "digits",
        "music_wrapped",
        "qr",
        "trainer",
        "transcript",
        "turkic",
    ]
    assert [member.value for member in EventSuffix] == [
        "started",
        "progress",
        "completed",
        "failed",
    ]
    assert [member.value for member in ErrorKind] == ["user", "system"]


def test_roundtrip_started_event() -> None:
    event = make_started_event(domain=JobDomain.TURKIC, job_id="j1", user_id=5, queue="primary")
    decoded = _roundtrip(event)
    assert decoded == event
    assert decoded["domain"] is JobDomain.TURKIC


def test_roundtrip_started_event_qr_domain() -> None:
    event = make_started_event(domain=JobDomain.QR, job_id="q1", user_id=9, queue="qr-queue")
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_progress_event_with_message() -> None:
    event = make_progress_event(
        domain=JobDomain.DIGITS,
        job_id="j2",
        user_id=7,
        progress=42,
        message="working",
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_progress_event_with_payload() -> None:
    event = make_progress_event(
        domain=JobDomain.DIGITS,
        job_id="j2",
        user_id=7,
        progress=42,
        message="working",
        payload={"metrics": {"loss": 0.1}},
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_progress_event_without_message() -> None:
    event = make_progress_event(
        domain=JobDomain.TRAINER, job_id="j3", user_id=1, progress=99, message=None
    )
    decoded = _roundtrip(event)
    assert decoded == {
        "type": "trainer.job.progress.v1",
        "domain": JobDomain.TRAINER,
        "job_id": "j3",
        "user_id": 1,
        "progress": 99,
    }


def test_roundtrip_completed_event() -> None:
    event = make_completed_event(
        domain=JobDomain.DATABANK,
        job_id="j4",
        user_id=11,
        result_id="file-1",
        result_bytes=2048,
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_roundtrip_failed_event_user_kind() -> None:
    event = make_failed_event(
        domain=JobDomain.TRANSCRIPT,
        job_id="j5",
        user_id=13,
        error_kind=ErrorKind.USER,
        message="invalid input",
    )
    decoded = _roundtrip(event)
    assert decoded == event


def test_encoded_event_carries_the_plain_words() -> None:
    encoded = encode_job_event(
        make_failed_event(
            domain=JobDomain.TURKIC,
            job_id="j7",
            user_id=3,
            error_kind=ErrorKind.SYSTEM,
            message="m",
        )
    )
    assert '"domain":"turkic"' in encoded.replace(" ", "")
    assert '"error_kind":"system"' in encoded.replace(" ", "")


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ('{"type": 1}', "Field 'type' must be a string, got int"),
        ('{"type": "x.y.z"}', "Invalid job event type format: 'x.y.z'"),
        (
            '{"type": "unknown.job.started.v1"}',
            "Invalid domain 'unknown': must be one of 'covenant', 'databank', 'digits', "
            "'music_wrapped', 'qr', 'trainer', 'transcript', 'turkic'",
        ),
        (
            '{"type": "turkic.job.invalid.v1"}',
            "Invalid event suffix 'invalid': must be one of "
            "'started', 'progress', 'completed', 'failed'",
        ),
        (
            '{"type": "turkic.job.started.v1", "domain": 1}',
            "Field 'domain' must be a string, got int",
        ),
        (
            '{"type": "turkic.job.started.v1", "domain": "transcript"}',
            "Job event domain mismatch: type says 'turkic', field says 'transcript'",
        ),
        (
            '{"type": "turkic.job.started.v1", "domain": "turkic", "job_id": 1, "user_id": 2}',
            "Field 'job_id' must be a string, got int",
        ),
        (
            '{"type": "turkic.job.started.v1", "domain": "turkic", "job_id": "j", "user_id": 1}',
            "Missing required field 'queue'",
        ),
        (
            '{"type": "turkic.job.progress.v1", "domain": "turkic", "job_id": "j", "user_id": 1}',
            "Missing required field 'progress'",
        ),
        (
            '{"type": "turkic.job.completed.v1", "domain": "turkic", "job_id": "j", "user_id": 1}',
            "Missing required field 'result_id'",
        ),
        (
            '{"type": "turkic.job.failed.v1", "domain": "turkic", '
            '"job_id": "j", "user_id": 1, "message": "x"}',
            "Missing required field 'error_kind'",
        ),
        (
            '{"type": "turkic.job.failed.v1", "domain": "turkic", '
            '"job_id": "j", "user_id": 1, "error_kind": "oops", "message": "x"}',
            "Invalid error_kind 'oops': must be one of 'user', 'system'",
        ),
    ],
)
def test_decode_job_event_invalid(payload: str, expected_message: str) -> None:
    with pytest.raises(JSONTypeError) as excinfo:
        decode_job_event(payload)
    assert str(excinfo.value) == expected_message


def test_decode_job_event_requires_object() -> None:
    with pytest.raises(JSONTypeError):
        decode_job_event("[]")


def test_failed_event_system_kind() -> None:
    payload = encode_job_event(
        make_failed_event(
            domain=JobDomain.TURKIC,
            job_id="j6",
            user_id=22,
            error_kind=ErrorKind.SYSTEM,
            message="boom",
        )
    )
    decoded = decode_job_event(payload)
    assert _is_failed(decoded)
    assert decoded["error_kind"] is ErrorKind.SYSTEM


def test_is_started_typeguard() -> None:
    started = make_started_event(domain=JobDomain.TURKIC, job_id="j1", user_id=1, queue="q")
    progress = make_progress_event(domain=JobDomain.TURKIC, job_id="j2", user_id=1, progress=50)
    completed = make_completed_event(
        domain=JobDomain.TURKIC, job_id="j3", user_id=1, result_id="r", result_bytes=100
    )
    failed = make_failed_event(
        domain=JobDomain.TURKIC,
        job_id="j4",
        user_id=1,
        error_kind=ErrorKind.SYSTEM,
        message="err",
    )
    assert is_started(started) is True
    assert is_started(progress) is False
    assert is_started(completed) is False
    assert is_started(failed) is False


def test_is_progress_typeguard() -> None:
    started = make_started_event(domain=JobDomain.TURKIC, job_id="j1", user_id=1, queue="q")
    progress = make_progress_event(domain=JobDomain.TURKIC, job_id="j2", user_id=1, progress=50)
    completed = make_completed_event(
        domain=JobDomain.TURKIC, job_id="j3", user_id=1, result_id="r", result_bytes=100
    )
    failed = make_failed_event(
        domain=JobDomain.TURKIC,
        job_id="j4",
        user_id=1,
        error_kind=ErrorKind.SYSTEM,
        message="err",
    )
    assert is_progress(started) is False
    assert is_progress(progress) is True
    assert is_progress(completed) is False
    assert is_progress(failed) is False


def test_is_completed_typeguard() -> None:
    started = make_started_event(domain=JobDomain.TURKIC, job_id="j1", user_id=1, queue="q")
    progress = make_progress_event(domain=JobDomain.TURKIC, job_id="j2", user_id=1, progress=50)
    completed = make_completed_event(
        domain=JobDomain.TURKIC, job_id="j3", user_id=1, result_id="r", result_bytes=100
    )
    failed = make_failed_event(
        domain=JobDomain.TURKIC,
        job_id="j4",
        user_id=1,
        error_kind=ErrorKind.SYSTEM,
        message="err",
    )
    assert is_completed(started) is False
    assert is_completed(progress) is False
    assert is_completed(completed) is True
    assert is_completed(failed) is False


def test_is_failed_typeguard() -> None:
    started = make_started_event(domain=JobDomain.TURKIC, job_id="j1", user_id=1, queue="q")
    progress = make_progress_event(domain=JobDomain.TURKIC, job_id="j2", user_id=1, progress=50)
    completed = make_completed_event(
        domain=JobDomain.TURKIC, job_id="j3", user_id=1, result_id="r", result_bytes=100
    )
    failed = make_failed_event(
        domain=JobDomain.TURKIC,
        job_id="j4",
        user_id=1,
        error_kind=ErrorKind.SYSTEM,
        message="err",
    )
    assert is_failed(started) is False
    assert is_failed(progress) is False
    assert is_failed(completed) is False
    assert is_failed(failed) is True


def test_covenant_domain_event_type_and_channel() -> None:
    assert make_event_type(JobDomain.COVENANT, EventSuffix.STARTED) == "covenant.job.started.v1"
    assert default_events_channel(JobDomain.COVENANT) == "covenant:events"


def test_roundtrip_covenant_started_event() -> None:
    event = make_started_event(
        domain=JobDomain.COVENANT, job_id="cov1", user_id=42, queue="covenant"
    )
    decoded = _roundtrip(event)
    assert decoded == event
