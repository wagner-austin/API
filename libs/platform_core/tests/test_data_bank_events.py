from __future__ import annotations

import pytest

from platform_core.data_bank_events import (
    DEFAULT_DATA_BANK_EVENTS_CHANNEL,
    CompletedV1,
    FailedV1,
    ProgressV1,
    StartedV1,
    decode_event,
    encode_event,
    is_completed,
    is_failed,
    is_progress,
    is_started,
)
from platform_core.json_utils import JSONTypeError, dump_json_str


def test_default_channel() -> None:
    assert DEFAULT_DATA_BANK_EVENTS_CHANNEL == "data_bank:events"


def test_encode_decode_started() -> None:
    started: StartedV1 = {
        "type": "data_bank.job.started.v1",
        "job_id": "j1",
        "user_id": 42,
        "queue": "data_bank",
    }
    decoded = decode_event(encode_event(started))
    assert is_started(decoded)
    assert decoded["job_id"] == "j1"
    assert decoded["user_id"] == 42
    assert decoded["queue"] == "data_bank"


def test_encode_decode_progress_with_message() -> None:
    progress: ProgressV1 = {
        "type": "data_bank.job.progress.v1",
        "job_id": "j1",
        "user_id": 42,
        "progress": 10,
        "message": "ok",
    }
    decoded = decode_event(encode_event(progress))
    assert is_progress(decoded)
    assert decoded["progress"] == 10
    assert decoded.get("message") == "ok"


def test_encode_decode_progress_without_message() -> None:
    progress_min: ProgressV1 = {
        "type": "data_bank.job.progress.v1",
        "job_id": "j1",
        "user_id": 42,
        "progress": 0,
    }
    decoded = decode_event(encode_event(progress_min))
    assert is_progress(decoded)
    assert decoded["progress"] == 0
    assert "message" not in decoded


def test_encode_decode_completed() -> None:
    done: CompletedV1 = {
        "type": "data_bank.job.completed.v1",
        "job_id": "j1",
        "user_id": 42,
        "file_id": "fid-1",
        "upload_status": "uploaded",
    }
    decoded = decode_event(encode_event(done))
    assert is_completed(decoded)
    assert decoded["file_id"] == "fid-1"
    assert decoded["upload_status"] == "uploaded"


def test_encode_decode_failed_system() -> None:
    failed: FailedV1 = {
        "type": "data_bank.job.failed.v1",
        "job_id": "j1",
        "user_id": 42,
        "error_kind": "system",
        "message": "boom",
    }
    decoded = decode_event(encode_event(failed))
    assert is_failed(decoded)
    assert decoded["error_kind"] == "system"
    assert decoded["message"] == "boom"


def test_encode_decode_failed_user() -> None:
    failed_user: FailedV1 = {
        "type": "data_bank.job.failed.v1",
        "job_id": "j2",
        "user_id": 42,
        "error_kind": "user",
        "message": "bad input",
    }
    decoded = decode_event(encode_event(failed_user))
    assert is_failed(decoded)
    assert decoded["error_kind"] == "user"


def test_decode_raises_for_non_object() -> None:
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        decode_event("[]")


def test_decode_raises_for_invalid_json() -> None:
    from platform_core.json_utils import InvalidJsonError

    with pytest.raises(InvalidJsonError):
        decode_event("not-json")


def test_decode_raises_for_non_string_type() -> None:
    bad_type: dict[str, str | int] = {"type": 123, "job_id": "j", "user_id": 1}
    with pytest.raises(JSONTypeError, match="Field 'type' must be a string"):
        decode_event(dump_json_str(bad_type))


def test_decode_raises_for_missing_queue_in_started() -> None:
    d_started_missing: dict[str, str | int] = {
        "type": "data_bank.job.started.v1",
        "job_id": "j",
        "user_id": 1,
    }
    with pytest.raises(JSONTypeError, match="Missing required field 'queue'"):
        decode_event(dump_json_str(d_started_missing))


def test_decode_raises_for_missing_progress() -> None:
    d_progress_missing: dict[str, str | int] = {
        "type": "data_bank.job.progress.v1",
        "job_id": "j",
        "user_id": 1,
    }
    with pytest.raises(JSONTypeError, match="Missing required field 'progress'"):
        decode_event(dump_json_str(d_progress_missing))


def test_decode_raises_for_wrong_progress_type() -> None:
    d_progress_wrong: dict[str, str | int] = {
        "type": "data_bank.job.progress.v1",
        "job_id": "j",
        "user_id": 1,
        "progress": "10",
    }
    with pytest.raises(JSONTypeError, match="Field 'progress' must be an integer"):
        decode_event(dump_json_str(d_progress_wrong))


def test_decode_raises_for_wrong_upload_status() -> None:
    d_completed_bad: dict[str, str | int] = {
        "type": "data_bank.job.completed.v1",
        "job_id": "j",
        "user_id": 1,
        "file_id": "f",
        "upload_status": "nope",
    }
    with pytest.raises(JSONTypeError, match="Invalid upload_status 'nope'"):
        decode_event(dump_json_str(d_completed_bad))


def test_decode_raises_for_invalid_error_kind() -> None:
    d_failed_bad: dict[str, str | int] = {
        "type": "data_bank.job.failed.v1",
        "job_id": "j",
        "user_id": 1,
        "error_kind": "oops",
        "message": "m",
    }
    with pytest.raises(JSONTypeError, match="Invalid error_kind 'oops'"):
        decode_event(dump_json_str(d_failed_bad))


def test_decode_raises_for_unknown_event_type() -> None:
    payload = '{"type":"data_bank.job.unknown.v1","job_id":"j","user_id":1}'
    with pytest.raises(JSONTypeError, match="Unknown data bank event type"):
        decode_event(payload)


def test_decode_handles_extra_keys() -> None:
    payload = (
        '{"extra_key":"x","type":"data_bank.job.completed.v1",'
        '"job_id":"j","user_id":42,"file_id":"f","upload_status":"uploaded"}'
    )
    decoded = decode_event(payload)
    assert is_completed(decoded)


def test_type_guards_return_false_for_non_matching() -> None:
    started: StartedV1 = {
        "type": "data_bank.job.started.v1",
        "job_id": "j",
        "user_id": 1,
        "queue": "q",
    }
    assert is_started(started)
    assert not is_progress(started)
    assert not is_completed(started)
    assert not is_failed(started)
