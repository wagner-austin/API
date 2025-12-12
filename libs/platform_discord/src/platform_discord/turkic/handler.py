"""Turkic event handler for decoding and routing turkic job events.

This module provides functions to decode and route events from the
turkic:events channel to the turkic runtime for Discord embed generation.

Events handled:
- turkic.job.started.v1 -> on_started()
- turkic.job.progress.v1 -> on_progress()
- turkic.job.completed.v1 -> on_completed()
- turkic.job.failed.v1 -> on_failed()
"""

from __future__ import annotations

from typing import TypeGuard

from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
    decode_job_event,
)
from platform_core.json_utils import InvalidJsonError, JSONTypeError
from platform_core.logging import get_logger

from .runtime import (
    RequestAction,
    TurkicRuntime,
    on_completed,
    on_failed,
    on_progress,
    on_started,
)

TurkicEventV1 = JobStartedV1 | JobProgressV1 | JobCompletedV1 | JobFailedV1

_logger = get_logger(__name__)


def _narrow_turkic(ev: JobEventV1) -> TurkicEventV1 | None:
    """Narrow JobEventV1 to TurkicEventV1 if type matches, else None."""
    if is_started(ev):
        return ev
    if is_progress(ev):
        return ev
    if is_completed(ev):
        return ev
    if is_failed(ev):
        return ev
    return None


def decode_turkic_event(payload: str) -> TurkicEventV1 | None:
    """Decode a turkic event from JSON payload.

    Returns None if the payload is not a recognized turkic event.
    """
    try:
        ev: JobEventV1 = decode_job_event(payload)
    except (InvalidJsonError, JSONTypeError):
        _logger.debug("Payload is not a recognized turkic event")
        return None
    if ev["domain"] != "turkic":
        return None
    return _narrow_turkic(ev)


def is_started(event: JobEventV1) -> TypeGuard[JobStartedV1]:
    """Check if the event is a started event."""
    return event.get("type") == "turkic.job.started.v1"


def is_progress(event: JobEventV1) -> TypeGuard[JobProgressV1]:
    """Check if the event is a progress event."""
    return event.get("type") == "turkic.job.progress.v1"


def is_completed(event: JobEventV1) -> TypeGuard[JobCompletedV1]:
    """Check if the event is a completed event."""
    return event.get("type") == "turkic.job.completed.v1"


def is_failed(event: JobEventV1) -> TypeGuard[JobFailedV1]:
    """Check if the event is a failed event."""
    return event.get("type") == "turkic.job.failed.v1"


def handle_turkic_event(
    runtime: TurkicRuntime,
    event: TurkicEventV1,
) -> RequestAction | None:
    """Route a decoded event to the appropriate runtime handler.

    Returns RequestAction with embed if the event should trigger a notification,
    or None if the event should be silently ignored.
    """
    if is_started(event):
        return on_started(
            runtime,
            user_id=event["user_id"],
            job_id=event["job_id"],
            queue=event["queue"],
        )

    if is_progress(event):
        return on_progress(
            runtime,
            user_id=event["user_id"],
            job_id=event["job_id"],
            progress=event["progress"],
            message=event.get("message"),
        )

    if is_completed(event):
        return on_completed(
            runtime,
            user_id=event["user_id"],
            job_id=event["job_id"],
            result_id=event["result_id"],
            result_bytes=event["result_bytes"],
        )

    if is_failed(event):
        return on_failed(
            runtime,
            user_id=event["user_id"],
            job_id=event["job_id"],
            error_kind=event["error_kind"],
            message=event["message"],
            status="failed",
        )

    return None


__all__ = [
    "TurkicEventV1",
    "_narrow_turkic",
    "decode_turkic_event",
    "handle_turkic_event",
    "is_completed",
    "is_failed",
    "is_progress",
    "is_started",
]
