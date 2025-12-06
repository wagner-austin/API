"""Trainer event handler for decoding job_events + trainer_metrics_events.

This module provides functions to decode and route events from the
trainer:events channel to the trainer runtime for Discord embed generation.

Events handled:
- trainer.metrics.config.v1 -> on_config()
- trainer.metrics.progress.v1 -> on_progress()
- trainer.metrics.completed.v1 -> on_completed()
- trainer.job.failed.v1 -> on_failed()
"""

from __future__ import annotations

from typing import TypeGuard

from platform_core.job_events import (
    JobEventV1,
    JobFailedV1,
    decode_job_event,
    is_failed,
)
from platform_core.logging import get_logger
from platform_core.trainer_metrics_events import (
    TrainerCompletedMetricsV1,
    TrainerConfigV1,
    TrainerMetricsEventV1,
    TrainerProgressMetricsV1,
    decode_trainer_metrics_event,
)

from .runtime import (
    RequestAction,
    TrainerRuntime,
    on_completed,
    on_config,
    on_failed,
    on_progress,
)

TrainerEventV1 = TrainerMetricsEventV1 | JobFailedV1

_logger = get_logger(__name__)


def decode_trainer_event(payload: str) -> TrainerEventV1 | None:
    """Decode a trainer event from JSON payload.

    Attempts to decode as trainer metrics event first, then as job event.
    Returns None if the payload is not a recognized trainer event.
    """
    metrics_event = _try_decode_metrics(payload)
    if metrics_event is not None:
        return metrics_event
    return _try_decode_job_failed(payload)


def _try_decode_metrics(payload: str) -> TrainerMetricsEventV1 | None:
    """Attempt to decode as trainer metrics event, returning None on failure."""
    try:
        return decode_trainer_metrics_event(payload)
    except ValueError:
        _logger.debug("Payload is not a trainer metrics event, trying job event")
        return None


def _try_decode_job_failed(payload: str) -> JobFailedV1 | None:
    """Attempt to decode as failed job event, returning None on failure."""
    try:
        ev: JobEventV1 = decode_job_event(payload)
        # Only handle failed events from job_events, and only for trainer domain
        if is_failed(ev) and ev["domain"] == "trainer":
            return ev
        return None
    except ValueError:
        _logger.debug("Payload is not a recognized trainer event")
        return None


def is_config(event: TrainerEventV1) -> TypeGuard[TrainerConfigV1]:
    """Check if the event is a config event."""
    return event.get("type") == "trainer.metrics.config.v1"


def is_progress(event: TrainerEventV1) -> TypeGuard[TrainerProgressMetricsV1]:
    """Check if the event is a progress metrics event."""
    return event.get("type") == "trainer.metrics.progress.v1"


def is_completed(event: TrainerEventV1) -> TypeGuard[TrainerCompletedMetricsV1]:
    """Check if the event is a completed metrics event."""
    return event.get("type") == "trainer.metrics.completed.v1"


def is_job_failed(event: TrainerEventV1) -> TypeGuard[JobFailedV1]:
    """Check if event is a JobFailedV1 by checking type field pattern."""
    type_field = event.get("type", "")
    return isinstance(type_field, str) and ".job.failed." in type_field


def handle_trainer_event(
    runtime: TrainerRuntime,
    event: TrainerEventV1,
) -> RequestAction | None:
    """Route a decoded event to the appropriate runtime handler.

    Returns RequestAction with embed if the event should trigger a notification,
    or None if the event should be silently ignored.
    """
    if is_config(event):
        return on_config(runtime, event)

    if is_progress(event):
        return on_progress(runtime, event)

    if is_completed(event):
        return on_completed(runtime, event)

    if is_job_failed(event):
        return on_failed(
            runtime,
            user_id=event["user_id"],
            request_id=event["job_id"],
            error_kind=event["error_kind"],
            message=event["message"],
            status="failed",
        )

    return None


__all__ = [
    "TrainerEventV1",
    "decode_trainer_event",
    "handle_trainer_event",
    "is_completed",
    "is_config",
    "is_job_failed",
    "is_progress",
]
