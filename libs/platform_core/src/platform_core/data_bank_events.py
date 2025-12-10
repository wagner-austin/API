from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal, TypedDict, TypeGuard

from .json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

DEFAULT_DATA_BANK_EVENTS_CHANNEL: Final[str] = "data_bank:events"


class StartedV1(TypedDict):
    type: Literal["data_bank.job.started.v1"]
    job_id: str
    user_id: int
    queue: str


class ProgressV1(TypedDict, total=False):
    type: Literal["data_bank.job.progress.v1"]
    job_id: str
    user_id: int
    progress: int
    message: str | None


class CompletedV1(TypedDict):
    type: Literal["data_bank.job.completed.v1"]
    job_id: str
    user_id: int
    file_id: str
    upload_status: Literal["uploaded"]


class FailedV1(TypedDict):
    type: Literal["data_bank.job.failed.v1"]
    job_id: str
    user_id: int
    error_kind: Literal["user", "system"]
    message: str


EventV1 = StartedV1 | ProgressV1 | CompletedV1 | FailedV1


def encode_event(event: EventV1) -> str:
    return dump_json_str(event)


def _decode_started(obj: JSONObject, job_id: str, user_id: int) -> StartedV1:
    queue = require_str(obj, "queue")
    return {
        "type": "data_bank.job.started.v1",
        "job_id": job_id,
        "user_id": user_id,
        "queue": queue,
    }


def _decode_progress(obj: JSONObject, job_id: str, user_id: int) -> ProgressV1:
    progress = require_int(obj, "progress")
    result: ProgressV1 = {
        "type": "data_bank.job.progress.v1",
        "job_id": job_id,
        "user_id": user_id,
        "progress": progress,
    }
    message = obj.get("message")
    if isinstance(message, str):
        result["message"] = message
    return result


def _decode_completed(obj: JSONObject, job_id: str, user_id: int) -> CompletedV1:
    file_id = require_str(obj, "file_id")
    upload_status = require_str(obj, "upload_status")
    if upload_status != "uploaded":
        raise JSONTypeError(f"Invalid upload_status '{upload_status}', expected 'uploaded'")
    return {
        "type": "data_bank.job.completed.v1",
        "job_id": job_id,
        "user_id": user_id,
        "file_id": file_id,
        "upload_status": "uploaded",
    }


def _decode_failed(obj: JSONObject, job_id: str, user_id: int) -> FailedV1:
    error_kind_raw = require_str(obj, "error_kind")
    message = require_str(obj, "message")
    if error_kind_raw == "user":
        error_kind: Literal["user", "system"] = "user"
    elif error_kind_raw == "system":
        error_kind = "system"
    else:
        raise JSONTypeError(f"Invalid error_kind '{error_kind_raw}' in failed event")
    return {
        "type": "data_bank.job.failed.v1",
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": error_kind,
        "message": message,
    }


_DECODERS: dict[str, Callable[[JSONObject, str, int], EventV1]] = {
    "data_bank.job.started.v1": _decode_started,
    "data_bank.job.progress.v1": _decode_progress,
    "data_bank.job.completed.v1": _decode_completed,
    "data_bank.job.failed.v1": _decode_failed,
}


def decode_event(payload: str) -> EventV1:
    """Parse and validate a serialized data bank event.

    Raises:
        JSONTypeError: if the payload is not a well-formed data bank event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))

    type_raw = require_str(decoded, "type")
    job_id = require_str(decoded, "job_id")
    user_id = require_int(decoded, "user_id")

    decoder = _DECODERS.get(type_raw)
    if decoder is None:
        raise JSONTypeError(f"Unknown data bank event type: '{type_raw}'")
    return decoder(decoded, job_id, user_id)


def is_started(ev: EventV1) -> TypeGuard[StartedV1]:
    return ev["type"] == "data_bank.job.started.v1"


def is_progress(ev: EventV1) -> TypeGuard[ProgressV1]:
    return ev["type"] == "data_bank.job.progress.v1"


def is_completed(ev: EventV1) -> TypeGuard[CompletedV1]:
    return ev["type"] == "data_bank.job.completed.v1"


def is_failed(ev: EventV1) -> TypeGuard[FailedV1]:
    return ev["type"] == "data_bank.job.failed.v1"


__all__ = [
    "DEFAULT_DATA_BANK_EVENTS_CHANNEL",
    "CompletedV1",
    "EventV1",
    "FailedV1",
    "ProgressV1",
    "StartedV1",
    "decode_event",
    "encode_event",
    "is_completed",
    "is_failed",
    "is_progress",
    "is_started",
]
