from __future__ import annotations

from _pytest.monkeypatch import MonkeyPatch

from platform_core.data_bank_events import (
    DEFAULT_DATA_BANK_EVENTS_CHANNEL,
    CompletedV1,
    EventV1,
    FailedV1,
    ProgressV1,
    StartedV1,
    encode_event,
    is_completed,
    is_failed,
    is_progress,
    is_started,
    try_decode_event,
)
from platform_core.json_utils import dump_json_str


def test_encode_decode_and_typeguards() -> None:
    assert DEFAULT_DATA_BANK_EVENTS_CHANNEL == "data_bank:events"

    started: StartedV1 = {
        "type": "data_bank.job.started.v1",
        "job_id": "j1",
        "user_id": 42,
        "queue": "data_bank",
    }
    dec0 = try_decode_event(encode_event(started))
    assert dec0 is not None and is_started(dec0)

    progress: ProgressV1 = {
        "type": "data_bank.job.progress.v1",
        "job_id": "j1",
        "user_id": 42,
        "progress": 10,
        "message": "ok",
    }
    dec1 = try_decode_event(encode_event(progress))
    assert dec1 is not None and is_progress(dec1)

    # Progress without optional message exercises the alternate branch
    progress_min: ProgressV1 = {
        "type": "data_bank.job.progress.v1",
        "job_id": "j1",
        "user_id": 42,
        "progress": 0,
    }
    dec1b = try_decode_event(encode_event(progress_min))
    assert dec1b is not None and is_progress(dec1b)

    done: CompletedV1 = {
        "type": "data_bank.job.completed.v1",
        "job_id": "j1",
        "user_id": 42,
        "file_id": "fid-1",
        "upload_status": "uploaded",
    }
    dec2 = try_decode_event(encode_event(done))
    assert dec2 is not None and is_completed(dec2)

    failed: FailedV1 = {
        "type": "data_bank.job.failed.v1",
        "job_id": "j1",
        "user_id": 42,
        "error_kind": "system",
        "message": "boom",
    }
    dec3 = try_decode_event(encode_event(failed))
    assert dec3 is not None and is_failed(dec3)

    # Test user error_kind for completeness
    failed_user: FailedV1 = {
        "type": "data_bank.job.failed.v1",
        "job_id": "j2",
        "user_id": 42,
        "error_kind": "user",
        "message": "bad input",
    }
    dec4 = try_decode_event(encode_event(failed_user))
    assert dec4 is not None and is_failed(dec4)


def test_decode_invalid_and_edge_paths() -> None:
    # not json object
    assert try_decode_event("[]") is None
    assert try_decode_event("not-json") is None
    bad_type: dict[str, str | int] = {"type": 123}
    assert try_decode_event(dump_json_str(bad_type)) is None

    # started: missing queue
    d_started_missing: dict[str, str | int] = {
        "type": "data_bank.job.started.v1",
        "job_id": "j",
    }
    assert try_decode_event(dump_json_str(d_started_missing)) is None

    # progress: missing progress or wrong type
    d_progress_missing: dict[str, str | int] = {
        "type": "data_bank.job.progress.v1",
        "job_id": "j",
    }
    assert try_decode_event(dump_json_str(d_progress_missing)) is None
    d_progress_wrong: dict[str, str | int] = {
        "type": "data_bank.job.progress.v1",
        "job_id": "j",
        "progress": "10",
    }
    assert try_decode_event(dump_json_str(d_progress_wrong)) is None

    # completed: wrong upload status
    d_completed_bad: dict[str, str | int] = {
        "type": "data_bank.job.completed.v1",
        "job_id": "j",
        "file_id": "f",
        "upload_status": "nope",
    }
    assert try_decode_event(dump_json_str(d_completed_bad)) is None

    # failed: wrong kind
    d_failed_bad: dict[str, str | int] = {
        "type": "data_bank.job.failed.v1",
        "job_id": "j",
        "error_kind": "oops",
        "message": "m",
    }
    assert try_decode_event(dump_json_str(d_failed_bad)) is None


def test_try_decode_non_string_keys() -> None:
    # JSON with extra key "1" - tests decoder handles unexpected keys gracefully
    payload = (
        '{"1":"x","type":"data_bank.job.completed.v1",'
        '"job_id":"j","user_id":42,"file_id":"f","upload_status":"uploaded"}'
    )
    out: EventV1 | None = try_decode_event(payload)
    assert out is not None and is_completed(out)


def test_data_bank_try_decode_json_loads_returns_non_dict(monkeypatch: MonkeyPatch) -> None:
    import platform_core.data_bank_events as mod
    import platform_core.json_utils as json_utils

    def _fake_loads(_s: str) -> list[str]:
        return ["not", "a", "dict"]

    monkeypatch.setattr(json_utils, "load_json_str", _fake_loads, raising=True)
    out = mod.try_decode_event("{}")
    assert out is None


def test_data_bank_try_decode_type_not_string_and_non_object_payload() -> None:
    assert try_decode_event("[]") is None
    assert try_decode_event('{"type":123,"job_id":"j"}') is None
    from platform_core.data_bank_events import _load_json_dict

    assert _load_json_dict("[]") is None


def test_data_bank_try_decode_handler_returns_none() -> None:
    invalid_payload = '{"type":"data_bank.job.started.v1","job_id":123,"ts":1,"queue":2}'
    assert try_decode_event(invalid_payload) is None


def test_data_bank_try_decode_parsed_none(monkeypatch: MonkeyPatch) -> None:
    import platform_core.data_bank_events as mod

    def _fake_load(_: str) -> list[str]:
        return ["not", "a", "dict"]

    monkeypatch.setattr(mod, "load_json_str", _fake_load, raising=True)
    assert mod.try_decode_event('{"type":"data_bank.job.started.v1"}') is None


def test_data_bank_unknown_event_type() -> None:
    payload = '{"type":"data_bank.job.unknown.v1","job_id":"j"}'
    assert try_decode_event(payload) is None
