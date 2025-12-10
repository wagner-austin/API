from __future__ import annotations

from typing import Literal

from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import JSONTypeError
from platform_core.turkic_jobs import TurkicJobStatus, turkic_job_key
from platform_workers.job_store import (
    BaseJobStore,
    JobStoreEncoder,
    parse_datetime_field,
    parse_int_field,
    parse_optional_str,
    parse_status,
)
from platform_workers.redis import RedisStrProto


class _TurkicJobEncoder(JobStoreEncoder[TurkicJobStatus]):
    def encode(self, status: TurkicJobStatus) -> dict[str, str]:
        return {
            "user_id": str(status["user_id"]),
            "status": status["status"],
            "progress": str(status["progress"]),
            "message": status["message"] or "",
            "result_url": status["result_url"] or "",
            "file_id": status["file_id"] or "",
            "upload_status": status["upload_status"] or "",
            "created_at": status["created_at"].isoformat(),
            "updated_at": status["updated_at"].isoformat(),
            "error": status["error"] or "",
        }

    def decode(self, job_id: str, raw: dict[str, str]) -> TurkicJobStatus:
        return {
            "job_id": job_id,
            "user_id": parse_int_field(raw, "user_id"),
            "status": parse_status(raw),
            "progress": parse_int_field(raw, "progress"),
            "message": parse_optional_str(raw, "message"),
            "result_url": parse_optional_str(raw, "result_url"),
            "file_id": parse_optional_str(raw, "file_id"),
            "upload_status": _parse_upload_status(raw),
            "created_at": parse_datetime_field(raw, "created_at"),
            "updated_at": parse_datetime_field(raw, "updated_at"),
            "error": parse_optional_str(raw, "error"),
        }


def _parse_upload_status(raw: dict[str, str]) -> Literal["uploaded"] | None:
    """Parse upload_status field from Redis hash.

    Raises:
        JSONTypeError: If upload_status is not None, empty, or 'uploaded'.
    """
    upload_raw = raw.get("upload_status")
    if upload_raw is None or upload_raw == "":
        return None
    if upload_raw == "uploaded":
        return "uploaded"
    raise JSONTypeError("invalid upload_status in redis store")


class TurkicJobStore:
    def __init__(self, redis: RedisStrProto) -> None:
        self._store = BaseJobStore[TurkicJobStatus](
            redis=redis,
            domain="turkic",
            encoder=_TurkicJobEncoder(),
        )
        self._redis = redis

    def save(self, status: TurkicJobStatus) -> None:
        self._store.save(status)

    def load(self, job_id: str) -> TurkicJobStatus | None:
        return self._store.load(job_id)

    def save_upload_metadata(self, job_id: str, meta: FileUploadResponse) -> None:
        key = f"{turkic_job_key(job_id)}:file"
        self._redis.hset(key, _encode_upload_meta(meta))

    def load_upload_metadata(self, job_id: str) -> FileUploadResponse:
        """Load upload metadata from Redis.

        Raises:
            JSONTypeError: If metadata is missing or malformed.
        """
        key = f"{turkic_job_key(job_id)}:file"
        raw = self._redis.hgetall(key)
        if not raw:
            raise JSONTypeError("upload metadata missing")
        return _decode_upload_meta(raw)


def _encode_upload_meta(meta: FileUploadResponse) -> dict[str, str]:
    return {
        "file_id": meta["file_id"],
        "size": str(meta["size"]),
        "sha256": meta["sha256"],
        "content_type": meta["content_type"],
        "created_at": meta["created_at"] or "",
    }


def _decode_upload_meta(raw: dict[str, str]) -> FileUploadResponse:
    """Decode upload metadata from Redis hash.

    Raises:
        JSONTypeError: If any required field is missing or invalid.
    """
    file_id = raw.get("file_id")
    size_raw = raw.get("size")
    sha256 = raw.get("sha256")
    content_type = raw.get("content_type")
    created_at_raw = raw.get("created_at")

    if not isinstance(file_id, str) or file_id.strip() == "":
        raise JSONTypeError("missing or invalid file_id in upload metadata")
    if not isinstance(size_raw, str) or not size_raw.strip().isdigit():
        raise JSONTypeError("missing or invalid size in upload metadata")
    if not isinstance(sha256, str) or sha256.strip() == "":
        raise JSONTypeError("missing or invalid sha256 in upload metadata")
    if not isinstance(content_type, str) or content_type.strip() == "":
        raise JSONTypeError("missing or invalid content_type in upload metadata")
    if created_at_raw is None:
        raise JSONTypeError("missing created_at in upload metadata")

    created_at: str | None
    if created_at_raw == "":
        created_at = None
    elif isinstance(created_at_raw, str) and created_at_raw.strip() != "":
        created_at = created_at_raw
    else:
        raise JSONTypeError("missing or invalid created_at in upload metadata")

    return {
        "file_id": file_id.strip(),
        "size": int(size_raw),
        "sha256": sha256.strip(),
        "content_type": content_type.strip(),
        "created_at": created_at,
    }
