from __future__ import annotations

from typing import Final, Literal, TypedDict, TypeGuard

from .json_utils import JSONValue, dump_json_str, load_json_str

DEFAULT_DATA_BANK_EVENTS_CHANNEL: Final[str] = "data_bank:events"


def _load_json_dict(s: str) -> dict[str, JSONValue] | None:
    raw_value: JSONValue = load_json_str(s)
    raw: JSONValue
    raw = raw_value
    if not isinstance(raw, dict):
        return None
    return raw


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


def _decode_started(obj: dict[str, JSONValue]) -> StartedV1 | None:
    job_id = obj.get("job_id")
    user_id = obj.get("user_id")
    queue = obj.get("queue")
    if isinstance(job_id, str) and isinstance(user_id, int) and isinstance(queue, str):
        return {
            "type": "data_bank.job.started.v1",
            "job_id": job_id,
            "user_id": user_id,
            "queue": queue,
        }
    return None


def _decode_progress(obj: dict[str, JSONValue]) -> ProgressV1 | None:
    job_id = obj.get("job_id")
    user_id = obj.get("user_id")
    progress = obj.get("progress")
    message = obj.get("message")
    if isinstance(job_id, str) and isinstance(user_id, int) and isinstance(progress, int):
        result: ProgressV1 = {
            "type": "data_bank.job.progress.v1",
            "job_id": job_id,
            "user_id": user_id,
            "progress": progress,
        }
        if isinstance(message, str):
            result["message"] = message
        return result
    return None


def _decode_completed(obj: dict[str, JSONValue]) -> CompletedV1 | None:
    job_id = obj.get("job_id")
    user_id = obj.get("user_id")
    file_id = obj.get("file_id")
    upload_status = obj.get("upload_status")
    if (
        isinstance(job_id, str)
        and isinstance(user_id, int)
        and isinstance(file_id, str)
        and upload_status == "uploaded"
    ):
        return {
            "type": "data_bank.job.completed.v1",
            "job_id": job_id,
            "user_id": user_id,
            "file_id": file_id,
            "upload_status": "uploaded",
        }
    return None


def _decode_failed(obj: dict[str, JSONValue]) -> FailedV1 | None:
    job_id = obj.get("job_id")
    user_id = obj.get("user_id")
    kind = obj.get("error_kind")
    msg = obj.get("message")
    if (
        isinstance(job_id, str)
        and isinstance(user_id, int)
        and isinstance(kind, str)
        and kind in ("user", "system")
        and isinstance(msg, str)
    ):
        error_kind: Literal["user", "system"] = "user" if kind == "user" else "system"
        return {
            "type": "data_bank.job.failed.v1",
            "job_id": job_id,
            "user_id": user_id,
            "error_kind": error_kind,
            "message": msg,
        }
    return None


def try_decode_event(payload: str) -> EventV1 | None:
    s = payload.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    parsed = _load_json_dict(s)
    if parsed is None:
        return None
    typ = parsed.get("type")
    if typ == "data_bank.job.started.v1":
        return _decode_started(parsed)
    if typ == "data_bank.job.progress.v1":
        return _decode_progress(parsed)
    if typ == "data_bank.job.completed.v1":
        return _decode_completed(parsed)
    if typ == "data_bank.job.failed.v1":
        return _decode_failed(parsed)
    return None


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
    "encode_event",
    "is_completed",
    "is_failed",
    "is_progress",
    "is_started",
    "try_decode_event",
]
