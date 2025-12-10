from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

from platform_core.digits_metrics_events import (
    DEFAULT_DIGITS_EVENTS_CHANNEL,
    DigitsArtifactV1,
    DigitsBatchMetricsV1,
    DigitsBestMetricsV1,
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    DigitsEventV1,
    DigitsPruneV1,
    DigitsUploadV1,
    decode_digits_event,
)
from platform_core.job_events import (
    JobEventV1,
    JobFailedV1,
    decode_job_event,
    default_events_channel,
)
from platform_core.json_utils import InvalidJsonError, JSONTypeError
from platform_discord.trainer.handler import TrainerEventV1, decode_trainer_event

# Event data can be a dict with string keys, or None if decode fails
EventValue = str | int | float | bool | None | list[str | int | float | bool | None]
EventData = dict[str, EventValue] | None

DigitsEvent = (
    DigitsConfigV1
    | DigitsBatchMetricsV1
    | DigitsEpochMetricsV1
    | DigitsBestMetricsV1
    | DigitsArtifactV1
    | DigitsUploadV1
    | DigitsPruneV1
    | DigitsCompletedMetricsV1
    | JobFailedV1
    | None
)
TrainerEvent = TrainerEventV1 | None
TranscriptEvent = JobEventV1 | None


class ServiceDef(TypedDict):
    id: str
    channel: str
    decode_event: Callable[[str], DigitsEvent | TrainerEvent | TranscriptEvent]


def _decode_transcript(payload: str) -> JobEventV1 | None:
    ev = decode_job_event(payload)
    return ev if ev["domain"] == "transcript" else None


def _decode_trainer_safe(payload: str) -> TrainerEventV1 | None:
    """Decode trainer event, returning None on failure."""
    return decode_trainer_event(payload)


def _decode_digits_safe(payload: str) -> DigitsEventV1 | None:
    """Decode digits event, returning None on decode failure."""
    try:
        return decode_digits_event(payload)
    except (InvalidJsonError, JSONTypeError) as exc:
        from platform_core.logging import get_logger

        get_logger(__name__).debug("Failed to decode digits event: %s", exc)
        return None


SERVICE_REGISTRY: dict[str, ServiceDef] = {
    "digits": {
        "id": "digits",
        "channel": DEFAULT_DIGITS_EVENTS_CHANNEL,
        "decode_event": _decode_digits_safe,
    },
    "trainer": {
        "id": "trainer",
        "channel": default_events_channel("trainer"),
        "decode_event": _decode_trainer_safe,
    },
    "transcript": {
        "id": "transcript",
        "channel": default_events_channel("transcript"),
        "decode_event": _decode_transcript,
    },
}


__all__ = ["SERVICE_REGISTRY", "ServiceDef"]
