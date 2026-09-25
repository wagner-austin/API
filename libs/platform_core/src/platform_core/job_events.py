"""Generic job lifecycle events: started, progress, completed, failed.

Three closed vocabularies live here as :class:`enum.StrEnum`, one definition
each whose members are the same strings on the wire: :class:`JobDomain` (which
service's queue emitted the event), :class:`EventSuffix` (which lifecycle
step) and :class:`ErrorKind` (whose fault a failure was). They replaced
module-level ``Literal`` aliases, which the operator's "no type alias" covers
(MCPs board task 1374feba); an untrusted word narrows through
:mod:`platform_core.members`.

The four events form a discriminated union, written out at each use rather
than bound to a module-level name for the same reason.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import NotRequired, TypedDict, TypeGuard

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
from platform_core.members import as_member, require_member


class JobDomain(StrEnum):
    """The service whose job queue emitted an event."""

    COVENANT = "covenant"
    DATABANK = "databank"
    DIGITS = "digits"
    MUSIC_WRAPPED = "music_wrapped"
    QR = "qr"
    TRAINER = "trainer"
    TRANSCRIPT = "transcript"
    TURKIC = "turkic"


class EventSuffix(StrEnum):
    """The lifecycle step an event reports."""

    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ErrorKind(StrEnum):
    """Whose fault a failed job was: the caller's input or the system."""

    USER = "user"
    SYSTEM = "system"


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


def make_event_type(domain: JobDomain, suffix: EventSuffix) -> str:
    """Construct the canonical event type string."""
    return f"{domain.value}.job.{suffix.value}.v1"


def default_events_channel(domain: JobDomain) -> str:
    """Return the default Redis pub/sub channel for the domain."""
    return f"{domain.value}:events"


def encode_job_event(event: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1) -> str:
    """Serialize a job event to a compact JSON string."""
    return dump_json_str(event)


def make_started_event(*, domain: JobDomain, job_id: str, user_id: int, queue: str) -> JobStartedV1:
    """Create a started event."""
    return {
        "type": make_event_type(domain, EventSuffix.STARTED),
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
        "type": make_event_type(domain, EventSuffix.PROGRESS),
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
        "type": make_event_type(domain, EventSuffix.COMPLETED),
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
        "type": make_event_type(domain, EventSuffix.FAILED),
        "domain": domain,
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": error_kind,
        "message": message,
    }


def _parse_event_type(raw: str) -> tuple[JobDomain, EventSuffix]:
    segments = raw.split(".")
    if len(segments) != 4 or segments[1] != "job" or segments[3] != "v1":
        raise JSONTypeError(f"Invalid job event type format: '{raw}'")
    domain = as_member(segments[0], "domain", JobDomain)
    suffix = as_member(segments[2], "event suffix", EventSuffix)
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
    kind = require_member(decoded, "error_kind", ErrorKind)
    message = require_str(decoded, "message")
    return {
        "type": type_raw,
        "domain": domain_value,
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": kind,
        "message": message,
    }


_DECODERS: dict[
    EventSuffix,
    Callable[
        [str, JobDomain, str, int, JSONObject],
        JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
    ],
] = {
    EventSuffix.STARTED: _decode_started_event,
    EventSuffix.PROGRESS: _decode_progress_event,
    EventSuffix.COMPLETED: _decode_completed_event,
    EventSuffix.FAILED: _decode_failed_event,
}


def decode_job_event(payload: str) -> JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1:
    """Parse and validate a serialized job event.

    Raises:
        JSONTypeError: if the payload is not a well-formed job event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))

    type_raw = require_str(decoded, "type")
    domain, suffix = _parse_event_type(type_raw)

    domain_value = require_member(decoded, "domain", JobDomain)
    if domain_value is not domain:
        raise JSONTypeError(
            f"Job event domain mismatch: type says '{domain.value}', "
            f"field says '{domain_value.value}'"
        )

    job_id = require_str(decoded, "job_id")
    user_id = require_int(decoded, "user_id")

    return _DECODERS[suffix](type_raw, domain_value, job_id, user_id, decoded)


def is_started(
    ev: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> TypeGuard[JobStartedV1]:
    """Check if the event is a started event."""
    return ".job.started." in ev.get("type", "")


def is_progress(
    ev: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> TypeGuard[JobProgressV1]:
    """Check if the event is a progress event."""
    return ".job.progress." in ev.get("type", "")


def is_completed(
    ev: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> TypeGuard[JobCompletedV1]:
    """Check if the event is a completed event."""
    return ".job.completed." in ev.get("type", "")


def is_failed(
    ev: JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1,
) -> TypeGuard[JobFailedV1]:
    """Check if the event is a failed event."""
    return ".job.failed." in ev.get("type", "")


__all__ = [
    "ErrorKind",
    "EventSuffix",
    "JobCompletedV1",
    "JobDomain",
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
