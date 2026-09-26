"""API request and response models using TypedDict.

All models are TypedDict definitions with explicit parse functions for validation.
No Pydantic, no TYPE_CHECKING pattern - single source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from platform_core import job_types
from platform_core.errors import AppError
from platform_core.errors import ErrorCode as PlatformErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_bytes, load_json_str
from platform_core.members import as_member
from typing_extensions import TypedDict

from ..core.models import Language, Script, Source
from .validators import (
    _decode_bool,
    _decode_float_range,
    _decode_int_range,
    _decode_optional_literal,
    _decode_required_literal,
    _decode_str,
    _load_json_dict,
)

# TypedDict models


class JobCreate(TypedDict):
    """Request to create a new corpus extraction job."""

    user_id: int
    source: Source
    language: Language
    script: Script | None
    max_sentences: int
    transliterate: bool
    confidence_threshold: float


class JobResponse(TypedDict):
    """Response when creating a job."""

    job_id: str
    user_id: int
    status: job_types.JobStatus
    created_at: datetime


class JobStatus(TypedDict):
    """Full status of a job."""

    job_id: str
    user_id: int
    status: job_types.JobStatus
    progress: int
    message: str | None
    result_url: str | None
    file_id: str | None
    upload_status: Literal["uploaded"] | None
    created_at: datetime
    updated_at: datetime
    error: str | None


# Parse functions with explicit validation. The hook refuses a word outside
# the enum's values with the API's own 400, so as_member only narrows.


def _decode_source_literal(val: JSONValue) -> Source:
    decoded = _decode_required_literal(val, "source", frozenset(Source))
    return as_member(decoded, "source", Source)


def _decode_language_literal(val: JSONValue) -> Language:
    decoded = _decode_required_literal(val, "language", frozenset(Language))
    return as_member(decoded, "language", Language)


def _decode_script_literal(val: JSONValue) -> Script | None:
    decoded = _decode_optional_literal(val, "script", frozenset(Script))
    return None if decoded is None else as_member(decoded, "script", Script)


def _decode_job_create_from_unknown(payload: JSONValue) -> JobCreate:
    """Decode and validate JobCreate from unknown JSON.

    Uses explicit literal narrowing to satisfy mypy's strict TypedDict checking.
    """
    d = _load_json_dict(payload)

    user_id_raw: JSONValue = d.get("user_id")
    if not isinstance(user_id_raw, int):
        raise AppError(
            code=PlatformErrorCode.INVALID_INPUT,
            message="user_id must be an integer",
            http_status=400,
        )
    user_id: int = user_id_raw

    source_raw: JSONValue = d.get("source")
    source = _decode_source_literal(source_raw)

    language_raw: JSONValue = d.get("language")
    language = _decode_language_literal(language_raw)

    script = _decode_script_literal(d.get("script"))

    max_raw: JSONValue = d.get("max_sentences")
    max_sentences = _decode_int_range(
        max_raw,
        "max_sentences",
        ge=1,
        le=100000,
        default=1000,
    )

    transliterate_raw: JSONValue = d.get("transliterate")
    transliterate = _decode_bool(
        transliterate_raw,
        "transliterate",
        default=True,
    )

    confidence_raw: JSONValue = d.get("confidence_threshold")
    confidence_threshold = _decode_float_range(
        confidence_raw,
        "confidence_threshold",
        ge=0.0,
        le=1.0,
        default=0.95,
    )

    return {
        "user_id": user_id,
        "source": source,
        "language": language,
        "script": script,
        "max_sentences": max_sentences,
        "transliterate": transliterate,
        "confidence_threshold": confidence_threshold,
    }


def parse_job_create(body: bytes) -> JobCreate:
    """Parse and validate JobCreate from the raw request body (public API).

    Args:
        body: The request body exactly as received, UTF-8 encoded JSON.

    Returns:
        The validated job request.

    Raises:
        InvalidJsonError: When the body is not JSON.
        AppError: When the JSON is not an object or a field fails validation.
    """
    return _decode_job_create_from_unknown(load_json_bytes(body))


# JSON parsing helpers for tests
def parse_job_response_json(s: str) -> JobResponse:
    """Parse JobResponse from JSON string."""
    from datetime import datetime

    obj: JSONValue = load_json_str(s)
    if not isinstance(obj, dict):
        raise JSONTypeError("Expected JSON object")

    user_id_val = obj.get("user_id")
    if not isinstance(user_id_val, int):
        raise JSONTypeError("user_id must be an integer")

    status = as_member(_decode_str(obj.get("status"), "status"), "status", job_types.JobStatus)

    return {
        "job_id": _decode_str(obj.get("job_id"), "job_id"),
        "user_id": user_id_val,
        "status": status,
        "created_at": datetime.fromisoformat(_decode_str(obj.get("created_at"), "created_at")),
    }


def parse_job_status_json(s: str) -> JobStatus:
    """Parse JobStatus from JSON string."""
    from datetime import datetime

    obj: JSONValue = load_json_str(s)
    if not isinstance(obj, dict):
        raise JSONTypeError("Expected JSON object")

    user_id_val = obj.get("user_id")
    if not isinstance(user_id_val, int):
        raise JSONTypeError("user_id must be an integer")

    message_val = obj.get("message")
    message = _decode_str(message_val, "message") if message_val is not None else None

    result_url_val = obj.get("result_url")
    result_url = _decode_str(result_url_val, "result_url") if result_url_val is not None else None

    file_id_val = obj.get("file_id")
    file_id = _decode_str(file_id_val, "file_id") if file_id_val is not None else None

    upload_status_val = obj.get("upload_status")
    upload_status: Literal["uploaded"] | None
    if upload_status_val is None:
        upload_status = None
    else:
        upload_status_str = _decode_str(upload_status_val, "upload_status")
        upload_status = "uploaded" if upload_status_str == "uploaded" else None

    error_val = obj.get("error")
    error = _decode_str(error_val, "error") if error_val is not None else None

    status = as_member(_decode_str(obj.get("status"), "status"), "status", job_types.JobStatus)

    return {
        "job_id": _decode_str(obj.get("job_id"), "job_id"),
        "user_id": user_id_val,
        "status": status,
        "progress": _decode_int_range(obj.get("progress"), "progress", ge=0, le=100),
        "message": message,
        "result_url": result_url,
        "file_id": file_id,
        "upload_status": upload_status,
        "created_at": datetime.fromisoformat(_decode_str(obj.get("created_at"), "created_at")),
        "updated_at": datetime.fromisoformat(_decode_str(obj.get("updated_at"), "updated_at")),
        "error": error,
    }


__all__ = [
    "JobCreate",
    "JobResponse",
    "JobStatus",
    "parse_job_create",
    "parse_job_response_json",
    "parse_job_status_json",
]
