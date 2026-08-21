"""Decoding and narrowing for digits events (metrics + job lifecycle).

The event shapes and their factory functions live in
:mod:`platform_core.digits_metrics_events`; this module owns the wire
side the consumers use: payload decoding, the TypeGuard narrowers, and
the combined digits channel type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeGuard

from platform_core.digits_metrics_events import (
    DigitsArtifactV1,
    DigitsBatchMetricsV1,
    DigitsBestMetricsV1,
    DigitsCompletedMetricsV1,
    DigitsConfigV1,
    DigitsEpochMetricsV1,
    DigitsMetricsEventV1,
    DigitsPruneV1,
    DigitsUploadV1,
)
from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    JobStartedV1,
    default_events_channel,
)

from .json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)


def _decode_config_context(event: DigitsConfigV1, decoded: JSONObject) -> None:
    """Attach optional context fields to config event from decoded data."""
    cpu_cores = decoded.get("cpu_cores")
    if isinstance(cpu_cores, int) and not isinstance(cpu_cores, bool):
        event["cpu_cores"] = cpu_cores
    optimal_threads = decoded.get("optimal_threads")
    if isinstance(optimal_threads, int) and not isinstance(optimal_threads, bool):
        event["optimal_threads"] = optimal_threads
    memory_mb = decoded.get("memory_mb")
    if isinstance(memory_mb, int) and not isinstance(memory_mb, bool):
        event["memory_mb"] = memory_mb
    optimal_workers = decoded.get("optimal_workers")
    if isinstance(optimal_workers, int) and not isinstance(optimal_workers, bool):
        event["optimal_workers"] = optimal_workers
    max_batch_size = decoded.get("max_batch_size")
    if isinstance(max_batch_size, int) and not isinstance(max_batch_size, bool):
        event["max_batch_size"] = max_batch_size
    device = decoded.get("device")
    if isinstance(device, str):
        event["device"] = device


def _decode_config_augment(event: DigitsConfigV1, decoded: JSONObject) -> None:
    """Attach optional augmentation fields to config event from decoded data."""
    batch_size = decoded.get("batch_size")
    if isinstance(batch_size, int) and not isinstance(batch_size, bool):
        event["batch_size"] = batch_size
    learning_rate = decoded.get("learning_rate")
    if isinstance(learning_rate, int | float) and not isinstance(learning_rate, bool):
        event["learning_rate"] = float(learning_rate)
    augment = decoded.get("augment")
    if isinstance(augment, bool):
        event["augment"] = augment
    aug_rotate = decoded.get("aug_rotate")
    if isinstance(aug_rotate, int | float) and not isinstance(aug_rotate, bool):
        event["aug_rotate"] = float(aug_rotate)
    aug_translate = decoded.get("aug_translate")
    if isinstance(aug_translate, int | float) and not isinstance(aug_translate, bool):
        event["aug_translate"] = float(aug_translate)
    noise_prob = decoded.get("noise_prob")
    if isinstance(noise_prob, int | float) and not isinstance(noise_prob, bool):
        event["noise_prob"] = float(noise_prob)
    dots_prob = decoded.get("dots_prob")
    if isinstance(dots_prob, int | float) and not isinstance(dots_prob, bool):
        event["dots_prob"] = float(dots_prob)


def _decode_config_event(
    decoded: JSONObject,
    job_id: str,
    user_id: int,
) -> DigitsConfigV1:
    model_id = require_str(decoded, "model_id")
    total_epochs = require_int(decoded, "total_epochs")
    queue = require_str(decoded, "queue")
    event: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "total_epochs": total_epochs,
        "queue": queue,
    }
    _decode_config_context(event, decoded)
    _decode_config_augment(event, decoded)
    return event


def _decode_batch_metrics_event(
    decoded: JSONObject,
    job_id: str,
    user_id: int,
) -> DigitsBatchMetricsV1:
    model_id = require_str(decoded, "model_id")
    epoch = require_int(decoded, "epoch")
    total_epochs = require_int(decoded, "total_epochs")
    batch = require_int(decoded, "batch")
    total_batches = require_int(decoded, "total_batches")
    batch_loss = require_float(decoded, "batch_loss")
    batch_acc = require_float(decoded, "batch_acc")
    avg_loss = require_float(decoded, "avg_loss")
    samples_per_sec = require_float(decoded, "samples_per_sec")
    main_rss_mb = require_int(decoded, "main_rss_mb")
    workers_rss_mb = require_int(decoded, "workers_rss_mb")
    worker_count = require_int(decoded, "worker_count")
    cgroup_usage_mb = require_int(decoded, "cgroup_usage_mb")
    cgroup_limit_mb = require_int(decoded, "cgroup_limit_mb")
    cgroup_pct = require_float(decoded, "cgroup_pct")
    anon_mb = require_int(decoded, "anon_mb")
    file_mb = require_int(decoded, "file_mb")
    return {
        "type": "digits.metrics.batch.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "batch_loss": batch_loss,
        "batch_acc": batch_acc,
        "avg_loss": avg_loss,
        "samples_per_sec": samples_per_sec,
        "main_rss_mb": main_rss_mb,
        "workers_rss_mb": workers_rss_mb,
        "worker_count": worker_count,
        "cgroup_usage_mb": cgroup_usage_mb,
        "cgroup_limit_mb": cgroup_limit_mb,
        "cgroup_pct": cgroup_pct,
        "anon_mb": anon_mb,
        "file_mb": file_mb,
    }


def _decode_epoch_metrics_event(
    decoded: JSONObject,
    job_id: str,
    user_id: int,
) -> DigitsEpochMetricsV1:
    model_id = require_str(decoded, "model_id")
    epoch = require_int(decoded, "epoch")
    total_epochs = require_int(decoded, "total_epochs")
    train_loss = require_float(decoded, "train_loss")
    val_acc = require_float(decoded, "val_acc")
    time_s = require_float(decoded, "time_s")
    return {
        "type": "digits.metrics.epoch.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "time_s": time_s,
    }


def _decode_best_metrics_event(
    decoded: JSONObject,
    job_id: str,
    user_id: int,
) -> DigitsBestMetricsV1:
    model_id = require_str(decoded, "model_id")
    epoch = require_int(decoded, "epoch")
    val_acc = require_float(decoded, "val_acc")
    return {
        "type": "digits.metrics.best.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "epoch": epoch,
        "val_acc": val_acc,
    }


def _decode_artifact_event(
    decoded: JSONObject,
    job_id: str,
    user_id: int,
) -> DigitsArtifactV1:
    model_id = require_str(decoded, "model_id")
    path = require_str(decoded, "path")
    return {
        "type": "digits.metrics.artifact.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "path": path,
    }


def _decode_upload_event(
    decoded: JSONObject,
    job_id: str,
    user_id: int,
) -> DigitsUploadV1:
    model_id = require_str(decoded, "model_id")
    status = require_int(decoded, "status")
    model_bytes = require_int(decoded, "model_bytes")
    manifest_bytes = require_int(decoded, "manifest_bytes")
    file_id = require_str(decoded, "file_id")
    file_sha256 = require_str(decoded, "file_sha256")
    return {
        "type": "digits.metrics.upload.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "status": status,
        "model_bytes": model_bytes,
        "manifest_bytes": manifest_bytes,
        "file_id": file_id,
        "file_sha256": file_sha256,
    }


def _decode_prune_event(
    decoded: JSONObject,
    job_id: str,
    user_id: int,
) -> DigitsPruneV1:
    model_id = require_str(decoded, "model_id")
    deleted_count = require_int(decoded, "deleted_count")
    return {
        "type": "digits.metrics.prune.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "deleted_count": deleted_count,
    }


def _decode_completed_metrics_event(
    decoded: JSONObject,
    job_id: str,
    user_id: int,
) -> DigitsCompletedMetricsV1:
    model_id = require_str(decoded, "model_id")
    val_acc = require_float(decoded, "val_acc")
    return {
        "type": "digits.metrics.completed.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "val_acc": val_acc,
    }


_DECODERS: dict[
    str,
    Callable[[JSONObject, str, int], DigitsMetricsEventV1],
] = {
    "digits.metrics.config.v1": _decode_config_event,
    "digits.metrics.batch.v1": _decode_batch_metrics_event,
    "digits.metrics.epoch.v1": _decode_epoch_metrics_event,
    "digits.metrics.best.v1": _decode_best_metrics_event,
    "digits.metrics.artifact.v1": _decode_artifact_event,
    "digits.metrics.upload.v1": _decode_upload_event,
    "digits.metrics.prune.v1": _decode_prune_event,
    "digits.metrics.completed.v1": _decode_completed_metrics_event,
}


def decode_digits_metrics_event(payload: str) -> DigitsMetricsEventV1:
    """Parse and validate a serialized digits metrics event.

    Raises:
        JSONTypeError: if the payload is not a well-formed digits metrics event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))

    type_raw = require_str(decoded, "type")
    job_id = require_str(decoded, "job_id")
    user_id = require_int(decoded, "user_id")

    decoder = _DECODERS.get(type_raw)
    if decoder is None:
        raise JSONTypeError(f"Unknown digits metrics event type: '{type_raw}'")
    return decoder(decoded, job_id, user_id)


# -----------------------------------------------------------------------------
# TypeGuard functions for type narrowing
# -----------------------------------------------------------------------------


def is_config(ev: DigitsMetricsEventV1) -> TypeGuard[DigitsConfigV1]:
    """Check if the event is a config event."""
    return ev.get("type") == "digits.metrics.config.v1"


def is_batch(ev: DigitsMetricsEventV1) -> TypeGuard[DigitsBatchMetricsV1]:
    """Check if the event is a batch metrics event."""
    return ev.get("type") == "digits.metrics.batch.v1"


def is_epoch(ev: DigitsMetricsEventV1) -> TypeGuard[DigitsEpochMetricsV1]:
    """Check if the event is an epoch metrics event."""
    return ev.get("type") == "digits.metrics.epoch.v1"


def is_best(ev: DigitsMetricsEventV1) -> TypeGuard[DigitsBestMetricsV1]:
    """Check if the event is a best model metrics event."""
    return ev.get("type") == "digits.metrics.best.v1"


def is_artifact(ev: DigitsMetricsEventV1) -> TypeGuard[DigitsArtifactV1]:
    """Check if the event is an artifact event."""
    return ev.get("type") == "digits.metrics.artifact.v1"


def is_upload(ev: DigitsMetricsEventV1) -> TypeGuard[DigitsUploadV1]:
    """Check if the event is an upload event."""
    return ev.get("type") == "digits.metrics.upload.v1"


def is_prune(ev: DigitsMetricsEventV1) -> TypeGuard[DigitsPruneV1]:
    """Check if the event is a prune event."""
    return ev.get("type") == "digits.metrics.prune.v1"


def is_completed(ev: DigitsMetricsEventV1) -> TypeGuard[DigitsCompletedMetricsV1]:
    """Check if the event is a completed metrics event."""
    return ev.get("type") == "digits.metrics.completed.v1"


# -----------------------------------------------------------------------------
# Combined event type for digits channel (job lifecycle + domain metrics)
# -----------------------------------------------------------------------------

# Combined event type for digits channel
DigitsEventV1 = JobEventV1 | DigitsMetricsEventV1

# Default channel for digits events
DEFAULT_DIGITS_EVENTS_CHANNEL: str = default_events_channel("digits")


def _decode_job_started(decoded: JSONObject, job_id: str, user_id: int) -> JobStartedV1:
    """Decode a started event."""
    queue = require_str(decoded, "queue")
    return {
        "type": "digits.job.started.v1",
        "domain": "digits",
        "job_id": job_id,
        "user_id": user_id,
        "queue": queue,
    }


def _decode_job_completed(decoded: JSONObject, job_id: str, user_id: int) -> JobCompletedV1:
    """Decode a completed event."""
    result_id = require_str(decoded, "result_id")
    result_bytes = require_int(decoded, "result_bytes")
    return {
        "type": "digits.job.completed.v1",
        "domain": "digits",
        "job_id": job_id,
        "user_id": user_id,
        "result_id": result_id,
        "result_bytes": result_bytes,
    }


def _decode_job_failed(decoded: JSONObject, job_id: str, user_id: int) -> JobFailedV1:
    """Decode a failed event."""
    error_kind_raw = require_str(decoded, "error_kind")
    message = require_str(decoded, "message")
    if error_kind_raw == "user":
        error_kind: Literal["user", "system"] = "user"
    elif error_kind_raw == "system":
        error_kind = "system"
    else:
        raise JSONTypeError(f"Invalid error_kind '{error_kind_raw}' in failed event")
    return {
        "type": "digits.job.failed.v1",
        "domain": "digits",
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": error_kind,
        "message": message,
    }


_JOB_DECODERS: dict[str, Callable[[JSONObject, str, int], JobEventV1]] = {
    "digits.job.started.v1": _decode_job_started,
    "digits.job.completed.v1": _decode_job_completed,
    "digits.job.failed.v1": _decode_job_failed,
}


def decode_digits_event(payload: str) -> DigitsEventV1:
    """Parse and validate any event from the digits channel.

    Handles both job lifecycle events (digits.job.*.v1) and
    metrics events (digits.metrics.*.v1).

    Raises:
        JSONTypeError: if the payload is not a well-formed digits event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))

    type_raw = require_str(decoded, "type")
    job_id = require_str(decoded, "job_id")
    user_id = require_int(decoded, "user_id")

    # Check if it's a job lifecycle event (digits.job.*.v1)
    if type_raw.startswith("digits.job."):
        domain = require_str(decoded, "domain")
        if domain != "digits":
            raise JSONTypeError(f"Domain mismatch: expected 'digits', got '{domain}'")
        job_decoder = _JOB_DECODERS.get(type_raw)
        if job_decoder is None:
            raise JSONTypeError(f"Unknown digits job event type: '{type_raw}'")
        return job_decoder(decoded, job_id, user_id)

    # Check if it's a metrics event (digits.metrics.*.v1)
    if type_raw.startswith("digits.metrics."):
        metrics_decoder = _DECODERS.get(type_raw)
        if metrics_decoder is None:
            raise JSONTypeError(f"Unknown digits metrics event type: '{type_raw}'")
        return metrics_decoder(decoded, job_id, user_id)

    raise JSONTypeError(f"Unknown digits event type: '{type_raw}'")


# TypeGuard helpers for combined event type narrowing
def is_digits_job_started(ev: DigitsEventV1) -> TypeGuard[JobStartedV1]:
    """Check if a combined event is a job started event."""
    type_val = ev.get("type")
    return isinstance(type_val, str) and ".job.started." in type_val


def is_digits_job_completed(ev: DigitsEventV1) -> TypeGuard[JobCompletedV1]:
    """Check if a combined event is a job completed event."""
    type_val = ev.get("type")
    return isinstance(type_val, str) and ".job.completed." in type_val


def is_digits_job_failed(ev: DigitsEventV1) -> TypeGuard[JobFailedV1]:
    """Check if a combined event is a job failed event."""
    type_val = ev.get("type")
    return isinstance(type_val, str) and ".job.failed." in type_val


def is_digits_config(ev: DigitsEventV1) -> TypeGuard[DigitsConfigV1]:
    """Check if a combined event is a config event."""
    return ev.get("type") == "digits.metrics.config.v1"


def is_digits_batch(ev: DigitsEventV1) -> TypeGuard[DigitsBatchMetricsV1]:
    """Check if a combined event is a batch metrics event."""
    return ev.get("type") == "digits.metrics.batch.v1"


def is_digits_epoch(ev: DigitsEventV1) -> TypeGuard[DigitsEpochMetricsV1]:
    """Check if a combined event is an epoch metrics event."""
    return ev.get("type") == "digits.metrics.epoch.v1"


def is_digits_best(ev: DigitsEventV1) -> TypeGuard[DigitsBestMetricsV1]:
    """Check if a combined event is a best model metrics event."""
    return ev.get("type") == "digits.metrics.best.v1"


def is_digits_artifact(ev: DigitsEventV1) -> TypeGuard[DigitsArtifactV1]:
    """Check if a combined event is an artifact event."""
    return ev.get("type") == "digits.metrics.artifact.v1"


def is_digits_upload(ev: DigitsEventV1) -> TypeGuard[DigitsUploadV1]:
    """Check if a combined event is an upload event."""
    return ev.get("type") == "digits.metrics.upload.v1"


def is_digits_prune(ev: DigitsEventV1) -> TypeGuard[DigitsPruneV1]:
    """Check if a combined event is a prune event."""
    return ev.get("type") == "digits.metrics.prune.v1"


def is_digits_completed_metrics(ev: DigitsEventV1) -> TypeGuard[DigitsCompletedMetricsV1]:
    """Check if a combined event is a completed metrics event."""
    return ev.get("type") == "digits.metrics.completed.v1"


__all__ = [
    "DEFAULT_DIGITS_EVENTS_CHANNEL",
    "DigitsEventV1",
    "JobCompletedV1",
    "JobFailedV1",
    "JobStartedV1",
    "decode_digits_event",
    "decode_digits_metrics_event",
    "is_artifact",
    "is_batch",
    "is_best",
    "is_completed",
    "is_config",
    "is_digits_artifact",
    "is_digits_batch",
    "is_digits_best",
    "is_digits_completed_metrics",
    "is_digits_config",
    "is_digits_epoch",
    "is_digits_job_completed",
    "is_digits_job_failed",
    "is_digits_job_started",
    "is_digits_prune",
    "is_digits_upload",
    "is_epoch",
    "is_prune",
    "is_upload",
]
