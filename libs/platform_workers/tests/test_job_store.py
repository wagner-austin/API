from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from platform_core.job_types import BaseJobStatus, JobStatusLiteral

from platform_workers.job_store import (
    BaseJobStore,
    JobStoreEncoder,
    parse_datetime_field,
    parse_int_field,
    parse_optional_str,
    parse_status,
)
from platform_workers.testing import FakeRedis


class _SampleStatus(BaseJobStatus):
    note: str | None


class _SampleEncoder(JobStoreEncoder[_SampleStatus]):
    def encode(self, status: _SampleStatus) -> dict[str, str]:
        return {
            "user_id": str(status["user_id"]),
            "status": status["status"],
            "progress": str(status["progress"]),
            "message": status["message"] or "",
            "created_at": status["created_at"].isoformat(),
            "updated_at": status["updated_at"].isoformat(),
            "error": status["error"] or "",
            "note": status["note"] or "",
        }

    def decode(self, job_id: str, raw: dict[str, str]) -> _SampleStatus:
        return {
            "job_id": job_id,
            "user_id": parse_int_field(raw, "user_id"),
            "status": parse_status(raw),
            "progress": parse_int_field(raw, "progress"),
            "message": parse_optional_str(raw, "message"),
            "created_at": parse_datetime_field(raw, "created_at"),
            "updated_at": parse_datetime_field(raw, "updated_at"),
            "error": parse_optional_str(raw, "error"),
            "note": parse_optional_str(raw, "note"),
        }


def test_base_job_store_round_trip() -> None:
    redis = FakeRedis()
    encoder = _SampleEncoder()
    store: BaseJobStore[_SampleStatus] = BaseJobStore(
        redis=redis,
        domain="turkic",
        encoder=encoder,
    )
    now = datetime.utcnow()
    status: _SampleStatus = {
        "job_id": "job-1",
        "user_id": 9,
        "status": "processing",
        "progress": 10,
        "message": "started",
        "created_at": now,
        "updated_at": now + timedelta(seconds=5),
        "error": None,
        "note": "ready",
    }

    store.save(status)
    loaded = store.load("job-1")
    assert loaded == status
    assert store.load("missing") is None

    redis.assert_only_called({"hset", "expire", "hgetall"})


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"status": "unknown"}, "invalid status in redis store"),
        ({}, "missing status in redis store"),
    ],
)
def test_parse_status_errors(raw: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_status(raw)
    assert message in str(excinfo.value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("queued", "queued"),
        ("processing", "processing"),
        ("completed", "completed"),
        ("failed", "failed"),
    ],
)
def test_parse_status_success(value: str, expected: JobStatusLiteral) -> None:
    assert parse_status({"status": value}) == expected


def test_parse_int_field_validates() -> None:
    assert parse_int_field({"value": "10"}, "value") == 10
    assert parse_int_field({"value": "-5"}, "value") == -5
    with pytest.raises(ValueError):
        parse_int_field({}, "value")
    with pytest.raises(ValueError):
        parse_int_field({"value": "abc"}, "value")


def test_parse_datetime_field_validates() -> None:
    ts = datetime.utcnow().isoformat()
    assert parse_datetime_field({"created": ts}, "created") == datetime.fromisoformat(ts)
    with pytest.raises(ValueError):
        parse_datetime_field({}, "created")
    with pytest.raises(ValueError):
        parse_datetime_field({"created": ""}, "created")


def test_parse_optional_str_handles_missing_and_empty() -> None:
    assert parse_optional_str({}, "note") is None
    assert parse_optional_str({"note": ""}, "note") is None
    assert parse_optional_str({"note": " x "}, "note") == "x"
