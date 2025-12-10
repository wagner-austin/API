from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import (
    DEFAULT_DIGITS_EVENTS_CHANNEL,
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    decode_digits_event,
    encode_digits_metrics_event,
)
from platform_core.job_events import JobFailedV1
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str


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
    evt = decode_digits_event(s)
    assert evt["type"] == "digits.metrics.config.v1"
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
    evt2 = decode_digits_event(s2)
    assert evt2["type"] == "digits.metrics.epoch.v1"

    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r1",
        "user_id": 1,
        "model_id": "m",
        "val_acc": 0.97,
    }
    s3 = encode_digits_metrics_event(completed)
    evt3 = decode_digits_event(s3)
    assert evt3["type"] == "digits.metrics.completed.v1"

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
    assert evt4["type"] == "digits.job.failed.v1"


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
    evt = decode_digits_event(payload)
    assert "cpu_cores" not in evt and "max_batch_size" not in evt


def test_digits_events_decode_invalid_json_raises() -> None:
    with pytest.raises(InvalidJsonError):
        decode_digits_event("not json")


def test_digits_events_decode_non_dict_raises() -> None:
    with pytest.raises(JSONTypeError, match="Expected JSON object"):
        decode_digits_event("[]")


def test_digits_events_decode_started_missing_fields_raises() -> None:
    bad = {"type": "digits.metrics.config.v1", "job_id": "r", "user_id": 1, "model_id": "m"}
    with pytest.raises(JSONTypeError, match="Missing required field"):
        decode_digits_event(dump_json_str(bad))


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
    with pytest.raises(JSONTypeError, match="must be an integer"):
        decode_digits_event(dump_json_str(bad))


def test_digits_events_decode_completed_invalid_types_raises() -> None:
    bad = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "val_acc": "0.95",  # wrong type
    }
    with pytest.raises(JSONTypeError, match="must be a number"):
        decode_digits_event(dump_json_str(bad))


def test_digits_events_decode_unknown_type_raises() -> None:
    unknown = {"type": "foo", "job_id": "r", "user_id": 1, "model_id": "m"}
    with pytest.raises(JSONTypeError, match="Unknown digits event type"):
        decode_digits_event(dump_json_str(unknown))


def test_digits_events_decode_missing_type_raises() -> None:
    with pytest.raises(JSONTypeError, match="Missing required field 'type'"):
        decode_digits_event(dump_json_str({}))


def test_digits_events_decode_nonstring_type_raises() -> None:
    with pytest.raises(JSONTypeError, match="Field 'type' must be a string"):
        decode_digits_event(dump_json_str({"type": 123}))


def test_digits_events_decode_failed_invalid_error_kind() -> None:
    from platform_core.job_events import decode_job_event

    bad = {
        "type": "digits.job.failed.v1",
        "job_id": "r",
        "user_id": 1,
        "error_kind": "oops",  # invalid kind
        "message": "msg",
        "domain": "digits",
    }
    with pytest.raises(JSONTypeError):
        decode_job_event(dump_json_str(bad))


logger = logging.getLogger(__name__)
