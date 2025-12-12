"""Digits event handler for decoding and routing handwriting-ai training events.

This module provides functions to decode and route events from the
digits:events channel to the digits runtime for Discord embed generation.

Events handled:
- digits.metrics.config.v1 -> on_started()
- digits.metrics.batch.v1 -> on_batch()
- digits.metrics.epoch.v1 -> on_progress()
- digits.metrics.best.v1 -> on_best()
- digits.metrics.artifact.v1 -> on_artifact()
- digits.metrics.upload.v1 -> on_upload()
- digits.metrics.prune.v1 -> on_prune()
- digits.metrics.completed.v1 -> on_completed()
- digits.job.failed.v1 -> on_failed()
"""

from __future__ import annotations

from platform_core.digits_metrics_events import (
    DigitsArtifactV1,
    DigitsBatchMetricsV1,
    DigitsBestMetricsV1,
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    DigitsEventV1,
    DigitsPruneV1,
    DigitsUploadV1,
    JobFailedV1,
    decode_digits_event,
    is_digits_artifact,
    is_digits_batch,
    is_digits_best,
    is_digits_completed_metrics,
    is_digits_config,
    is_digits_epoch,
    is_digits_job_failed,
    is_digits_prune,
    is_digits_upload,
)
from platform_core.json_utils import InvalidJsonError, JSONTypeError
from platform_core.logging import get_logger

from .runtime import (
    DigitsRuntime,
    RequestAction,
    on_artifact,
    on_batch,
    on_best,
    on_completed,
    on_failed,
    on_progress,
    on_prune,
    on_started,
    on_upload,
)

_logger = get_logger(__name__)


def decode_digits_event_safe(payload: str) -> DigitsEventV1 | None:
    """Decode a digits event from JSON payload.

    Attempts to decode as digits event (job lifecycle or metrics).
    Returns None if the payload is not a recognized digits event.
    """
    try:
        return decode_digits_event(payload)
    except (InvalidJsonError, JSONTypeError):
        _logger.debug("Payload is not a recognized digits event")
        return None


def _extract_optional_int(event: DigitsConfigV1, key: str) -> int | None:
    """Extract optional int value from config event."""
    val = event.get(key)
    return val if isinstance(val, int) and not isinstance(val, bool) else None


def _extract_optional_str(event: DigitsConfigV1, key: str) -> str | None:
    """Extract optional str value from config event."""
    val = event.get(key)
    return val if isinstance(val, str) else None


def _extract_optional_float(event: DigitsConfigV1, key: str) -> float | None:
    """Extract optional float value from config event, converting int to float."""
    val = event.get(key)
    if isinstance(val, bool):
        return None
    if isinstance(val, int | float):
        return float(val)
    return None


def _extract_optional_bool(event: DigitsConfigV1, key: str) -> bool | None:
    """Extract optional bool value from config event."""
    val = event.get(key)
    return val if isinstance(val, bool) else None


def _handle_config(runtime: DigitsRuntime, event: DigitsConfigV1) -> RequestAction:
    """Handle config event (digits.metrics.config.v1)."""
    return on_started(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        model_id=event["model_id"],
        total_epochs=event["total_epochs"],
        queue=event["queue"],
        cpu_cores=_extract_optional_int(event, "cpu_cores"),
        optimal_threads=_extract_optional_int(event, "optimal_threads"),
        memory_mb=_extract_optional_int(event, "memory_mb"),
        optimal_workers=_extract_optional_int(event, "optimal_workers"),
        max_batch_size=_extract_optional_int(event, "max_batch_size"),
        device=_extract_optional_str(event, "device"),
        batch_size=_extract_optional_int(event, "batch_size"),
        learning_rate=_extract_optional_float(event, "learning_rate"),
        augment=_extract_optional_bool(event, "augment"),
        aug_rotate=_extract_optional_float(event, "aug_rotate"),
        aug_translate=_extract_optional_float(event, "aug_translate"),
        noise_prob=_extract_optional_float(event, "noise_prob"),
        dots_prob=_extract_optional_float(event, "dots_prob"),
    )


def _handle_batch(runtime: DigitsRuntime, event: DigitsBatchMetricsV1) -> RequestAction:
    """Handle batch metrics event (digits.metrics.batch.v1)."""
    return on_batch(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        model_id=event["model_id"],
        epoch=event["epoch"],
        total_epochs=event["total_epochs"],
        batch=event["batch"],
        total_batches=event["total_batches"],
        batch_loss=event["batch_loss"],
        batch_acc=event["batch_acc"],
        avg_loss=event["avg_loss"],
        samples_per_sec=event["samples_per_sec"],
        main_rss_mb=event["main_rss_mb"],
        workers_rss_mb=event["workers_rss_mb"],
        worker_count=event["worker_count"],
        cgroup_usage_mb=event["cgroup_usage_mb"],
        cgroup_limit_mb=event["cgroup_limit_mb"],
        cgroup_pct=event["cgroup_pct"],
        anon_mb=event["anon_mb"],
        file_mb=event["file_mb"],
    )


def _handle_epoch(runtime: DigitsRuntime, event: DigitsEpochMetricsV1) -> RequestAction:
    """Handle epoch metrics event (digits.metrics.epoch.v1)."""
    return on_progress(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        epoch=event["epoch"],
        total_epochs=event["total_epochs"],
        val_acc=event["val_acc"],
        train_loss=event["train_loss"],
        time_s=event["time_s"],
    )


def _handle_best(runtime: DigitsRuntime, event: DigitsBestMetricsV1) -> None:
    """Handle best model event (digits.metrics.best.v1)."""
    on_best(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        epoch=event["epoch"],
        val_acc=event["val_acc"],
    )


def _handle_artifact(runtime: DigitsRuntime, event: DigitsArtifactV1) -> None:
    """Handle artifact event (digits.metrics.artifact.v1)."""
    on_artifact(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        path=event["path"],
    )


def _handle_upload(runtime: DigitsRuntime, event: DigitsUploadV1) -> None:
    """Handle upload event (digits.metrics.upload.v1)."""
    on_upload(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        status=event["status"],
        model_bytes=event["model_bytes"],
        manifest_bytes=event["manifest_bytes"],
    )


def _handle_prune(runtime: DigitsRuntime, event: DigitsPruneV1) -> None:
    """Handle prune event (digits.metrics.prune.v1)."""
    on_prune(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        deleted_count=event["deleted_count"],
    )


def _handle_completed(
    runtime: DigitsRuntime, event: DigitsCompletedMetricsV1
) -> RequestAction | None:
    """Handle completed metrics event (digits.metrics.completed.v1)."""
    return on_completed(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        model_id=event["model_id"],
        run_id=None,
        val_acc=event["val_acc"],
    )


def _handle_failed(runtime: DigitsRuntime, event: JobFailedV1) -> RequestAction:
    """Handle job failed event (digits.job.failed.v1)."""
    return on_failed(
        runtime,
        user_id=event["user_id"],
        request_id=event["job_id"],
        model_id="unknown",
        error_kind=event["error_kind"],
        message=event["message"],
        queue="unknown",
        status="failed",
    )


def handle_digits_event(
    runtime: DigitsRuntime,
    event: DigitsEventV1,
) -> RequestAction | None:
    """Route a decoded event to the appropriate runtime handler.

    Returns RequestAction with embed if the event should trigger a notification,
    or None if the event should be silently ignored (best, artifact, upload, prune).
    """
    if is_digits_config(event):
        return _handle_config(runtime, event)

    if is_digits_batch(event):
        return _handle_batch(runtime, event)

    if is_digits_epoch(event):
        return _handle_epoch(runtime, event)

    if is_digits_best(event):
        _handle_best(runtime, event)
        return None

    if is_digits_artifact(event):
        _handle_artifact(runtime, event)
        return None

    if is_digits_upload(event):
        _handle_upload(runtime, event)
        return None

    if is_digits_prune(event):
        _handle_prune(runtime, event)
        return None

    if is_digits_completed_metrics(event):
        return _handle_completed(runtime, event)

    if is_digits_job_failed(event):
        return _handle_failed(runtime, event)

    return None


__all__ = [
    "DigitsEventV1",
    "decode_digits_event_safe",
    "handle_digits_event",
]
