from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal, NotRequired, TypedDict, TypeGuard

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

JobDomain = Literal[
    "covenant",
    "databank",
    "digits",
    "music_wrapped",
    "qr",
    "trainer",
    "transcript",
    "turkic",
]
EventSuffix = Literal["started", "progress", "completed", "failed"]
ErrorKind = Literal["user", "system"]

_DOMAIN_VALUES: Final[tuple[JobDomain, ...]] = (
    "covenant",
    "databank",
    "digits",
    "music_wrapped",
    "qr",
    "trainer",
    "transcript",
    "turkic",
)
_SUFFIX_VALUES: Final[tuple[EventSuffix, ...]] = ("started", "progress", "completed", "failed")


class JobStartedV1(TypedDict):
    """Generic job started event."""

    type: str
    domain: JobDomain
    job_id: str
    user_id: int
    queue: str


class JobProgressV1(TypedDict):
    """Generic job progress event."""

    type: str
    domain: JobDomain
    job_id: str
    user_id: int
    progress: int
    message: NotRequired[str]
    payload: NotRequired[JSONValue]


class JobCompletedV1(TypedDict):
    """Generic job completed event."""

    type: str
    domain: JobDomain
    job_id: str
    user_id: int
    result_id: str
    result_bytes: int


class JobFailedV1(TypedDict):
    """Generic job failed event."""

    type: str
    domain: JobDomain
    job_id: str
    user_id: int
    error_kind: ErrorKind
    message: str


JobEventV1 = JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1


def make_event_type(domain: JobDomain, suffix: EventSuffix) -> str:
    """Construct the canonical event type string."""
    return f"{domain}.job.{suffix}.v1"


def default_events_channel(domain: JobDomain) -> str:
    """Return the default Redis pub/sub channel for the domain."""
    return f"{domain}:events"


def encode_job_event(event: JobEventV1) -> str:
    """Serialize a job event to a compact JSON string."""
    return dump_json_str(event)


def make_started_event(*, domain: JobDomain, job_id: str, user_id: int, queue: str) -> JobStartedV1:
    """Create a started event."""
    return {
        "type": make_event_type(domain, "started"),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "queue": queue,
    }


def make_progress_event(
    *,
    domain: JobDomain,
    job_id: str,
    user_id: int,
    progress: int,
    message: str | None = None,
    payload: JSONValue | None = None,
) -> JobProgressV1:
    """Create a progress event."""
    event: JobProgressV1 = {
        "type": make_event_type(domain, "progress"),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "progress": progress,
    }
    if message is not None:
        event["message"] = message
    if payload is not None:
        event["payload"] = payload
    return event


def make_completed_event(
    *,
    domain: JobDomain,
    job_id: str,
    user_id: int,
    result_id: str,
    result_bytes: int,
) -> JobCompletedV1:
    """Create a completed event."""
    return {
        "type": make_event_type(domain, "completed"),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "result_id": result_id,
        "result_bytes": result_bytes,
    }


def make_failed_event(
    *, domain: JobDomain, job_id: str, user_id: int, error_kind: ErrorKind, message: str
) -> JobFailedV1:
    """Create a failed event."""
    return {
        "type": make_event_type(domain, "failed"),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": error_kind,
        "message": message,
    }


def _parse_domain(raw: str) -> JobDomain:
    for value in _DOMAIN_VALUES:
        if raw == value:
            return value
    raise JSONTypeError(f"Invalid domain '{raw}' in job event")


def _parse_suffix(raw: str) -> EventSuffix:
    for value in _SUFFIX_VALUES:
        if raw == value:
            return value
    raise JSONTypeError(f"Invalid event suffix '{raw}' in job event")


def _parse_event_type(raw: str) -> tuple[JobDomain, EventSuffix]:
    segments = raw.split(".")
    if len(segments) != 4 or segments[1] != "job" or segments[3] != "v1":
        raise JSONTypeError(f"Invalid job event type format: '{raw}'")
    domain_str, suffix_str = segments[0], segments[2]
    domain = _parse_domain(domain_str)
    suffix = _parse_suffix(suffix_str)
    return domain, suffix


def _decode_started_event(
    type_raw: str, domain_value: JobDomain, job_id: str, user_id: int, decoded: JSONObject
) -> JobStartedV1:
    queue = require_str(decoded, "queue")
    return {
        "type": type_raw,
        "domain": domain_value,
        "job_id": job_id,
        "user_id": user_id,
        "queue": queue,
    }


def _decode_progress_event(
    type_raw: str, domain_value: JobDomain, job_id: str, user_id: int, decoded: JSONObject
) -> JobProgressV1:
    progress = require_int(decoded, "progress")
    event: JobProgressV1 = {
        "type": type_raw,
        "domain": domain_value,
        "job_id": job_id,
        "user_id": user_id,
        "progress": progress,
    }
    message = decoded.get("message")
    if isinstance(message, str):
        event["message"] = message
    if "payload" in decoded:
        event["payload"] = decoded["payload"]
    return event


def _decode_completed_event(
    type_raw: str, domain_value: JobDomain, job_id: str, user_id: int, decoded: JSONObject
) -> JobCompletedV1:
    result_id = require_str(decoded, "result_id")
    result_bytes = require_int(decoded, "result_bytes")
    return {
        "type": type_raw,
        "domain": domain_value,
        "job_id": job_id,
        "user_id": user_id,
        "result_id": result_id,
        "result_bytes": result_bytes,
    }


def _decode_failed_event(
    type_raw: str, domain_value: JobDomain, job_id: str, user_id: int, decoded: JSONObject
) -> JobFailedV1:
    error_kind_raw = require_str(decoded, "error_kind")
    message = require_str(decoded, "message")
    if error_kind_raw == "user":
        kind: ErrorKind = "user"
    elif error_kind_raw == "system":
        kind = "system"
    else:
        raise JSONTypeError(f"Invalid error_kind '{error_kind_raw}' in failed event")
    return {
        "type": type_raw,
        "domain": domain_value,
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": kind,
        "message": message,
    }


def decode_job_event(payload: str) -> JobEventV1:
    """Parse and validate a serialized job event.

    Raises:
        JSONTypeError: if the payload is not a well-formed job event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))

    type_raw = require_str(decoded, "type")
    domain, suffix = _parse_event_type(type_raw)

    domain_field = require_str(decoded, "domain")
    domain_value = _parse_domain(domain_field)
    if domain_value != domain:
        raise JSONTypeError(
            f"Job event domain mismatch: type says '{domain}', field says '{domain_value}'"
        )

    job_id = require_str(decoded, "job_id")
    user_id = require_int(decoded, "user_id")

    decoder_type = Callable[[str, JobDomain, str, int, JSONObject], JobEventV1]
    decoders: dict[EventSuffix, decoder_type] = {
        "started": _decode_started_event,
        "progress": _decode_progress_event,
        "completed": _decode_completed_event,
        "failed": _decode_failed_event,
    }
    decoder = decoders[suffix]
    return decoder(type_raw, domain_value, job_id, user_id, decoded)


def is_started(ev: JobEventV1) -> TypeGuard[JobStartedV1]:
    """Check if the event is a started event."""
    return ".job.started." in ev.get("type", "")


def is_progress(ev: JobEventV1) -> TypeGuard[JobProgressV1]:
    """Check if the event is a progress event."""
    return ".job.progress." in ev.get("type", "")


def is_completed(ev: JobEventV1) -> TypeGuard[JobCompletedV1]:
    """Check if the event is a completed event."""
    return ".job.completed." in ev.get("type", "")


def is_failed(ev: JobEventV1) -> TypeGuard[JobFailedV1]:
    """Check if the event is a failed event."""
    return ".job.failed." in ev.get("type", "")


__all__ = [
    "ErrorKind",
    "EventSuffix",
    "JobCompletedV1",
    "JobDomain",
    "JobEventV1",
    "JobFailedV1",
    "JobProgressV1",
    "JobStartedV1",
    "decode_job_event",
    "default_events_channel",
    "encode_job_event",
    "is_completed",
    "is_failed",
    "is_progress",
    "is_started",
    "make_completed_event",
    "make_event_type",
    "make_failed_event",
    "make_progress_event",
    "make_started_event",
]
