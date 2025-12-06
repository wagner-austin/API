"""Redis-backed job store for transcript STT jobs."""

from __future__ import annotations

from datetime import datetime

from platform_core.job_types import JobStatusLiteral, job_key
from platform_workers.job_store import (
    BaseJobStore,
    JobStoreEncoder,
    parse_datetime_field,
    parse_int_field,
    parse_optional_str,
    parse_status,
)
from platform_workers.redis import RedisStrProto
from typing_extensions import TypedDict


def transcript_job_key(job_id: str) -> str:
    """Generate Redis key for transcript job status."""
    return job_key("transcript", job_id)


class TranscriptJobStatus(TypedDict):
    """Status of a transcript STT job."""

    job_id: str
    user_id: int
    status: JobStatusLiteral
    progress: int
    message: str | None
    url: str
    video_id: str | None
    text: str | None
    created_at: datetime
    updated_at: datetime
    error: str | None


class _TranscriptJobEncoder(JobStoreEncoder[TranscriptJobStatus]):
    def encode(self, status: TranscriptJobStatus) -> dict[str, str]:
        return {
            "user_id": str(status["user_id"]),
            "status": status["status"],
            "progress": str(status["progress"]),
            "message": status["message"] or "",
            "url": status["url"],
            "video_id": status["video_id"] or "",
            "text": status["text"] or "",
            "created_at": status["created_at"].isoformat(),
            "updated_at": status["updated_at"].isoformat(),
            "error": status["error"] or "",
        }

    def decode(self, job_id: str, raw: dict[str, str]) -> TranscriptJobStatus:
        url = raw.get("url")
        if not isinstance(url, str) or url.strip() == "":
            raise ValueError("missing url in redis store")

        return {
            "job_id": job_id,
            "user_id": parse_int_field(raw, "user_id"),
            "status": parse_status(raw),
            "progress": parse_int_field(raw, "progress"),
            "message": parse_optional_str(raw, "message"),
            "url": url,
            "video_id": parse_optional_str(raw, "video_id"),
            "text": parse_optional_str(raw, "text"),
            "created_at": parse_datetime_field(raw, "created_at"),
            "updated_at": parse_datetime_field(raw, "updated_at"),
            "error": parse_optional_str(raw, "error"),
        }


class TranscriptJobStore:
    """Redis-backed store for transcript job status."""

    def __init__(self, redis: RedisStrProto) -> None:
        self._store = BaseJobStore[TranscriptJobStatus](
            redis=redis,
            domain="transcript",
            encoder=_TranscriptJobEncoder(),
        )

    def save(self, status: TranscriptJobStatus) -> None:
        self._store.save(status)

    def load(self, job_id: str) -> TranscriptJobStatus | None:
        return self._store.load(job_id)


__all__ = [
    "JobStatusLiteral",
    "TranscriptJobStatus",
    "TranscriptJobStore",
    "transcript_job_key",
]
