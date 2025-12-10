from __future__ import annotations

from datetime import datetime
from typing import Generic, Protocol, TypeVar

from platform_core.job_types import BaseJobStatus, JobStatusLiteral, job_key
from platform_core.json_utils import JSONTypeError

from platform_workers.redis import RedisStrProto

TStatus = TypeVar("TStatus", bound=BaseJobStatus)


class JobStoreEncoder(Protocol[TStatus]):
    """Protocol for encoding/decoding job status structures."""

    def encode(self, status: TStatus) -> dict[str, str]: ...

    def decode(self, job_id: str, raw: dict[str, str]) -> TStatus: ...


class BaseJobStore(Generic[TStatus]):
    """Generic Redis-backed job store with strict parsing."""

    def __init__(
        self,
        *,
        redis: RedisStrProto,
        domain: str,
        encoder: JobStoreEncoder[TStatus],
    ) -> None:
        self._redis = redis
        self._domain = domain
        self._encoder = encoder

    def save(self, status: TStatus) -> None:
        key = job_key(self._domain, status["job_id"])
        self._redis.hset(key, mapping=self._encoder.encode(status))

    def load(self, job_id: str) -> TStatus | None:
        key = job_key(self._domain, job_id)
        raw = self._redis.hgetall(key)
        if not raw:
            return None
        return self._encoder.decode(job_id, raw)


def parse_status(raw: dict[str, str]) -> JobStatusLiteral:
    """Parse a status field from a Redis hash."""
    status_raw = raw.get("status")
    if status_raw is None:
        raise JSONTypeError("missing status in redis store")
    if status_raw == "queued":
        return "queued"
    if status_raw == "processing":
        return "processing"
    if status_raw == "completed":
        return "completed"
    if status_raw == "failed":
        return "failed"
    raise JSONTypeError("invalid status in redis store")


def parse_int_field(raw: dict[str, str], key: str) -> int:
    """Parse an integer field from a Redis hash."""
    value = raw.get(key)
    if value is None:
        raise JSONTypeError(f"missing {key} in redis store")
    stripped = value.strip()
    if stripped == "" or not stripped.lstrip("-").isdigit():
        raise JSONTypeError(f"invalid {key} in redis store")
    return int(stripped)


def parse_datetime_field(raw: dict[str, str], key: str) -> datetime:
    """Parse an ISO 8601 datetime field from a Redis hash."""
    value = raw.get(key)
    if value is None or value.strip() == "":
        raise JSONTypeError(f"missing {key} in redis store")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise JSONTypeError(f"invalid {key} in redis store: {exc}") from exc


def parse_optional_str(raw: dict[str, str], key: str) -> str | None:
    """Parse an optional string field from a Redis hash."""
    value = raw.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped != "" else None


__all__ = [
    "BaseJobStore",
    "JobStoreEncoder",
    "parse_datetime_field",
    "parse_int_field",
    "parse_optional_str",
    "parse_status",
]
