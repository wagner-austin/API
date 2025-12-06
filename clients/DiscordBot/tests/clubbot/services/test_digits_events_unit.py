from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import (
    DEFAULT_DIGITS_EVENTS_CHANNEL,
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    encode_digits_metrics_event,
    try_decode_digits_event,
)
from platform_core.job_events import JobFailedV1
from platform_core.json_utils import InvalidJsonError, dump_json_str


def test_digits_events_encode_decode_roundtrip() -> None:
    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 10,
        "queue": "digits",
        # Optional extras should round-trip when present
        "cpu_cores": 2,
        "optimal_threads": 2,
        "memory_mb": 953,
        "optimal_workers": 0,
        "max_batch_size": 64,
        "device": "cpu",
    }
    s = encode_digits_metrics_event(config)  # should be JSON
    evt = try_decode_digits_event(s)
    assert evt is not None and evt["type"] == "digits.metrics.config.v1"
    assert DEFAULT_DIGITS_EVENTS_CHANNEL == "digits:events"
    assert evt.get("cpu_cores") == 2
    assert evt.get("memory_mb") == 953
    assert evt.get("max_batch_size") == 64
    assert evt.get("device") == "cpu"

    epoch: DigitsEpochMetricsV1 = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "epoch": 3,
        "total_epochs": 10,
        "train_loss": 0.1,
        "val_acc": 0.95,
        "time_s": 1.2,
    }
    s2 = encode_digits_metrics_event(epoch)
    evt2 = try_decode_digits_event(s2)
    assert evt2 is not None and evt2["type"] == "digits.metrics.epoch.v1"

    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "val_acc": 0.97,
    }
    s3 = encode_digits_metrics_event(completed)
    evt3 = try_decode_digits_event(s3)
    assert evt3 is not None and evt3["type"] == "digits.metrics.completed.v1"

    failed: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r1",
        "user_id": 1,
        "error_kind": "system",
        "message": "boom",
        "domain": "digits",
    }
    from platform_core.job_events import decode_job_event, encode_job_event

    s4 = encode_job_event(failed)
    evt4 = decode_job_event(s4)
    assert evt4 is not None and evt4["type"] == "digits.job.failed.v1"


def test_digits_events_decode_started_without_extras() -> None:
    payload = dump_json_str(
        {
            "type": "digits.metrics.config.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            "total_epochs": 5,
            "queue": "digits",
        }
    )
    evt = try_decode_digits_event(payload)
    assert evt is not None and "cpu_cores" not in evt and "max_batch_size" not in evt


def test_digits_events_decode_invalid_json_raises() -> None:
    with pytest.raises(InvalidJsonError):
        try_decode_digits_event("not json")


def test_digits_events_decode_non_dict_returns_none() -> None:
    assert try_decode_digits_event("[]") is None


def test_digits_events_decode_started_missing_fields_raises() -> None:
    bad = {"type": "digits.metrics.config.v1", "job_id": "r", "user_id": 1, "model_id": "m"}
    with pytest.raises(ValueError, match="config event requires"):
        try_decode_digits_event(dump_json_str(bad))


def test_digits_events_decode_progress_invalid_types_raises() -> None:
    bad = {
        "type": "digits.metrics.epoch.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "epoch": "3",  # wrong type
        "total_epochs": 10,
        "train_loss": 0.1,
        "val_acc": 0.5,
        "time_s": 1.0,
    }
    with pytest.raises(ValueError, match="epoch metrics event missing required fields"):
        try_decode_digits_event(dump_json_str(bad))


def test_digits_events_decode_completed_invalid_types_raises() -> None:
    bad = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "val_acc": "0.95",  # wrong type
    }
    with pytest.raises(ValueError, match="completed metrics event missing required fields"):
        try_decode_digits_event(dump_json_str(bad))


def test_digits_events_decode_unknown_type() -> None:
    unknown = {"type": "foo", "job_id": "r", "user_id": 1, "model_id": "m"}
    assert try_decode_digits_event(dump_json_str(unknown)) is None


def test_digits_events_decode_missing_type_or_nonstring() -> None:
    assert try_decode_digits_event(dump_json_str({})) is None
    assert try_decode_digits_event(dump_json_str({"type": 123})) is None


def test_digits_events_decode_failed_invalid_types() -> None:
    from platform_core.job_events import decode_job_event

    bad = {
        "type": "digits.job.failed.v1",
        "job_id": "r",
        "user_id": 1,
        "error_kind": "oops",  # invalid kind
        "message": "msg",
        "domain": "digits",
    }
    with pytest.raises(ValueError):
        decode_job_event(dump_json_str(bad))


def test_parse_json_obj_drops_non_string_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import platform_core.digits_metrics_events as mod
    import platform_core.json_utils as json_utils

    # Monkeypatch load_json_str to return a dict with a non-string key
    def _fake_loads(_s: str) -> dict[int | str, str]:
        return {1: "x", "type": "digits.metrics.completed.v1"}

    monkeypatch.setattr(json_utils, "load_json_str", _fake_loads, raising=True)
    # Unknown/incomplete payload should return None, exercising non-string key branch
    assert mod.try_decode_digits_event("{}") is None


logger = logging.getLogger(__name__)
