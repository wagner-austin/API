from __future__ import annotations

from datetime import datetime
from typing import Final

from platform_core.job_types import BaseJobStatus, JobStatusLiteral
from platform_workers.job_store import (
    BaseJobStore,
    JobStoreEncoder,
    parse_datetime_field,
    parse_int_field,
    parse_optional_str,
    parse_status,
)
from platform_workers.redis import RedisStrProto

_TRAINER_DOMAIN: Final[str] = "trainer"


class TrainerJobStatus(BaseJobStatus):
    """Persisted status for trainer jobs."""

    artifact_file_id: str | None


class _TrainerJobEncoder(JobStoreEncoder[TrainerJobStatus]):
    def encode(self, status: TrainerJobStatus) -> dict[str, str]:
        return {
            "user_id": str(status["user_id"]),
            "status": status["status"],
            "progress": str(status["progress"]),
            "message": status["message"] or "",
            "created_at": status["created_at"].isoformat(),
            "updated_at": status["updated_at"].isoformat(),
            "error": status["error"] or "",
            "artifact_file_id": status["artifact_file_id"] or "",
        }

    def decode(self, job_id: str, raw: dict[str, str]) -> TrainerJobStatus:
        return {
            "job_id": job_id,
            "user_id": parse_int_field(raw, "user_id"),
            "status": parse_status(raw),
            "progress": parse_int_field(raw, "progress"),
            "message": parse_optional_str(raw, "message"),
            "created_at": parse_datetime_field(raw, "created_at"),
            "updated_at": parse_datetime_field(raw, "updated_at"),
            "error": parse_optional_str(raw, "error"),
            "artifact_file_id": parse_optional_str(raw, "artifact_file_id"),
        }


class TrainerJobStore:
    """Typed Redis-backed store for trainer job statuses."""

    def __init__(self, redis: RedisStrProto) -> None:
        self._store = BaseJobStore[TrainerJobStatus](
            redis=redis,
            domain=_TRAINER_DOMAIN,
            encoder=_TrainerJobEncoder(),
        )

    def save(self, status: TrainerJobStatus) -> None:
        self._store.save(status)

    def load(self, job_id: str) -> TrainerJobStatus | None:
        return self._store.load(job_id)

    def initial_status(
        self,
        *,
        job_id: str,
        user_id: int,
        message: str,
        status: JobStatusLiteral,
    ) -> TrainerJobStatus:
        now = datetime.utcnow()
        return {
            "job_id": job_id,
            "user_id": user_id,
            "status": status,
            "progress": 0,
            "message": message,
            "created_at": now,
            "updated_at": now,
            "error": None,
            "artifact_file_id": None,
        }


__all__ = ["TrainerJobStatus", "TrainerJobStore"]
