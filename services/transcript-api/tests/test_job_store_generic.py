from __future__ import annotations

from datetime import datetime

import pytest

from transcript_api.job_store import TranscriptJobStatus, TranscriptJobStore, transcript_job_key


class _RedisStub:
    def __init__(self) -> None:
        self._hashes: dict[str, dict[str, str]] = {}

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self._hashes[key] = dict(mapping)
        return 1

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._hashes.get(key, {}))

    def publish(self, channel: str, message: str) -> int:
        return 1

    def ping(self, **kwargs: str | int | float | bool | None) -> bool:
        return True

    def set(self, key: str, value: str) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return None

    def sadd(self, key: str, *values: str) -> int:
        return len(values)

    def scard(self, key: str) -> int:
        return 0

    def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    def sismember(self, key: str, member: str) -> bool:
        return False

    def close(self) -> None:
        return None


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
    redis = _RedisStub()
    store = TranscriptJobStore(redis)
    status = _status("job-1")
    store.save(status)
    loaded = store.load("job-1")
    assert loaded == status
    assert transcript_job_key("job-1") in redis._hashes


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "unknown", "invalid status in redis store"),
        ("progress", "x", "invalid progress in redis store"),
        ("created_at", "", "missing created_at in redis store"),
        ("url", "", "missing url in redis store"),
    ],
)
def test_transcript_job_store_invalid_fields_raise(field: str, value: str, message: str) -> None:
    redis = _RedisStub()
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
    with pytest.raises(ValueError) as excinfo:
        store.load("bad")
    assert message in str(excinfo.value)
