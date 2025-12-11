"""API request and response models using TypedDict.

All models are TypedDict definitions with explicit parse functions for validation.
No Pydantic, no TYPE_CHECKING pattern - single source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, NotRequired

from platform_core.errors import AppError
from platform_core.errors import ErrorCode as PlatformErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str
from typing_extensions import TypedDict

from .types import JsonDict
from .validators import (
    _decode_bool,
    _decode_float_range,
    _decode_int_range,
    _decode_str,
    _load_json_dict,
)


def _hook_decode_required_literal(
    val: JSONValue,
    field: str,
    allowed: frozenset[str],
) -> str:
    """Decode required literal using hook for test injection."""
    from turkic_api import _test_hooks

    return _test_hooks.decode_required_literal(val, field, allowed)


def _hook_decode_optional_literal(
    val: JSONValue,
    field: str,
    allowed: frozenset[str],
) -> str | None:
    """Decode optional literal using hook for test injection."""
    from turkic_api import _test_hooks

    return _test_hooks.decode_optional_literal(val, field, allowed)


# Type aliases for literals
Status = Literal["queued", "processing", "completed", "failed"]
Script = Literal["Latn", "Cyrl", "Arab"]
Source = Literal["oscar", "wikipedia", "culturax"]
Language = Literal["kk", "ky", "uz", "tr", "ug", "fi", "az", "en"]
ErrorCode = Literal[
    "INVALID_REQUEST",
    "JOB_NOT_FOUND",
    "JOB_FAILED",
    "RATE_LIMIT_EXCEEDED",
    "INTERNAL_ERROR",
]

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
    status: Status
    created_at: datetime


class JobStatus(TypedDict):
    """Full status of a job."""

    job_id: str
    user_id: int
    status: Status
    progress: int
    message: str | None
    result_url: str | None
    file_id: str | None
    upload_status: Literal["uploaded"] | None
    created_at: datetime
    updated_at: datetime
    error: str | None


class ErrorResponse(TypedDict):
    """Error response."""

    error: str
    code: ErrorCode
    details: NotRequired[JsonDict]
    timestamp: datetime


# Parse functions with explicit validation

_SOURCE_VALUES = frozenset({"oscar", "wikipedia", "culturax"})
_LANGUAGE_VALUES = frozenset({"kk", "ky", "uz", "tr", "ug", "fi", "az", "en"})
_SCRIPT_VALUES = frozenset({"Latn", "Cyrl", "Arab"})
_SCRIPT_MAP: dict[str, Script] = {"Latn": "Latn", "Cyrl": "Cyrl", "Arab": "Arab"}


def _get_source_map() -> dict[str, str]:
    """Get source map from hooks for test injection."""
    from turkic_api import _test_hooks

    return _test_hooks.source_map


def _get_language_map() -> dict[str, str]:
    """Get language map from hooks for test injection."""
    from turkic_api import _test_hooks

    return _test_hooks.language_map


def _decode_source_literal(val: JSONValue) -> Source:
    decoded = _hook_decode_required_literal(val, "source", _SOURCE_VALUES)
    source_val = _get_source_map().get(decoded)
    if source_val is None:
        raise AppError(
            code=PlatformErrorCode.INVALID_INPUT,
            message="Invalid source",
            http_status=400,
        )
    # Narrow to Source literal
    if source_val == "oscar":
        return "oscar"
    if source_val == "wikipedia":
        return "wikipedia"
    if source_val == "culturax":
        return "culturax"
    raise AppError(
        code=PlatformErrorCode.INVALID_INPUT,
        message="Invalid source",
        http_status=400,
    )


def _decode_language_literal(val: JSONValue) -> Language:
    decoded = _hook_decode_required_literal(val, "language", _LANGUAGE_VALUES)
    lang_val = _get_language_map().get(decoded)
    if lang_val is None:
        raise AppError(
            code=PlatformErrorCode.INVALID_INPUT,
            message="Invalid language",
            http_status=400,
        )
    # Narrow to Language literal
    if lang_val == "kk":
        return "kk"
    if lang_val == "ky":
        return "ky"
    if lang_val == "uz":
        return "uz"
    if lang_val == "tr":
        return "tr"
    if lang_val == "ug":
        return "ug"
    if lang_val == "fi":
        return "fi"
    if lang_val == "az":
        return "az"
    if lang_val == "en":
        return "en"
    raise AppError(
        code=PlatformErrorCode.INVALID_INPUT,
        message="Invalid language",
        http_status=400,
    )


def _decode_script_literal(val: JSONValue) -> Script | None:
    if val is None:
        return None
    _hook_decode_optional_literal(val, "script", _SCRIPT_VALUES)
    if val == "Latn":
        return "Latn"
    if val == "Cyrl":
        return "Cyrl"
    if val == "Arab":
        return "Arab"
    return None


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

    script_raw: JSONValue = d.get("script")
    if script_raw is None:
        script: Script | None = None
    else:
        _hook_decode_optional_literal(script_raw, "script", _SCRIPT_VALUES)
        script = _SCRIPT_MAP.get(script_raw if isinstance(script_raw, str) else "")

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


def parse_job_create(payload: JsonDict) -> JobCreate:
    """Parse and validate JobCreate from request body (public API)."""
    converted: dict[str, JSONValue] = {}
    for k, v in payload.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            converted[k] = v
        else:
            # v must be a list given JsonDict type constraints
            items: list[JSONValue] = [
                item for item in v if item is None or isinstance(item, (str, int, float, bool))
            ]
            converted[k] = items
    converted_payload: JSONValue = converted
    return _decode_job_create_from_unknown(converted_payload)


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

    status_val = _decode_str(obj.get("status"), "status")
    if status_val == "queued":
        status: Status = "queued"
    elif status_val == "processing":
        status = "processing"
    elif status_val == "completed":
        status = "completed"
    elif status_val == "failed":
        status = "failed"
    else:
        raise JSONTypeError("Invalid job status")

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

    status_val = _decode_str(obj.get("status"), "status")
    if status_val == "queued":
        status: Status = "queued"
    elif status_val == "processing":
        status = "processing"
    elif status_val == "completed":
        status = "completed"
    elif status_val == "failed":
        status = "failed"
    else:
        raise JSONTypeError("Invalid job status")

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
    "ErrorCode",
    "ErrorResponse",
    "JobCreate",
    "JobResponse",
    "JobStatus",
    "Language",
    "Script",
    "Source",
    "Status",
    "parse_job_create",
    "parse_job_response_json",
    "parse_job_status_json",
]
