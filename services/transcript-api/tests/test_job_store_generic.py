from __future__ import annotations

from datetime import datetime

import pytest
from platform_core.json_utils import JSONTypeError
from platform_workers.testing import FakeRedis

from transcript_api.job_store import TranscriptJobStatus, TranscriptJobStore, transcript_job_key


def _status(job_id: str) -> TranscriptJobStatus:
    now = datetime.utcnow()
    return {
        "job_id": job_id,
        "user_id": 9,
        "status": "processing",
        "progress": 10,
        "message": None,
        "url": "https://youtu.be/abc",
        "video_id": "abc",
        "text": None,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }


def test_transcript_job_store_roundtrip() -> None:
    redis = FakeRedis()
    store = TranscriptJobStore(redis)
    status = _status("job-1")
    store.save(status)
    loaded = store.load("job-1")
    assert loaded == status
    assert transcript_job_key("job-1") in redis._hashes
    redis.assert_only_called({"hset", "expire", "hgetall"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "unknown", "invalid status in redis store"),
        ("progress", "x", "invalid progress in redis store"),
        ("created_at", "", "missing created_at in redis store"),
    ],
)
def test_transcript_job_store_invalid_fields_raise(field: str, value: str, message: str) -> None:
    redis = FakeRedis()
    now = datetime.utcnow().isoformat()
    redis.hset(
        transcript_job_key("bad"),
        {
            "user_id": "1",
            "status": "processing",
            "progress": "0",
            "message": "",
            "url": "https://youtu.be/x",
            "video_id": "",
            "text": "",
            "created_at": now,
            "updated_at": now,
            "error": "",
            field: value,
        },
    )
    store = TranscriptJobStore(redis)
    with pytest.raises(JSONTypeError) as excinfo:
        store.load("bad")
    assert message in str(excinfo.value)
    redis.assert_only_called({"hset", "expire", "hgetall"})


def test_transcript_job_store_missing_url_raises() -> None:
    redis = FakeRedis()
    now = datetime.utcnow().isoformat()
    redis.hset(
        transcript_job_key("bad"),
        {
            "user_id": "1",
            "status": "processing",
            "progress": "0",
            "message": "",
            "url": "",
            "video_id": "",
            "text": "",
            "created_at": now,
            "updated_at": now,
            "error": "",
        },
    )
    store = TranscriptJobStore(redis)
    with pytest.raises(JSONTypeError) as excinfo:
        store.load("bad")
    assert "missing url in redis store" in str(excinfo.value)
    redis.assert_only_called({"hset", "expire", "hgetall"})
