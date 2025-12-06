"""Digits training metrics events for Handwriting-AI.

This module provides TypedDict definitions and encoder/decoder functions
for domain-specific metrics events published during digits model training.

Lifecycle events (started, progress, completed, failed) are handled by
platform_workers.job_context via generic job_events.

Event types:
- digits.metrics.config.v1 -> Config (model settings, device, augmentation)
- digits.metrics.batch.v1 -> Batch progress (loss, acc, memory)
- digits.metrics.epoch.v1 -> Epoch completion (train_loss, val_acc)
- digits.metrics.best.v1 -> New best model found
- digits.metrics.artifact.v1 -> Artifact path saved
- digits.metrics.upload.v1 -> Upload status
- digits.metrics.prune.v1 -> Cleanup metrics
- digits.metrics.completed.v1 -> Final metrics (val_acc)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NotRequired, TypedDict, TypeGuard

from platform_core.job_events import (
    JobCompletedV1,
    JobEventV1,
    JobFailedV1,
    JobStartedV1,
    default_events_channel,
)

from .json_utils import JSONValue, dump_json_str, load_json_str

DigitsMetricsEventType = Literal[
    "digits.metrics.config.v1",
    "digits.metrics.batch.v1",
    "digits.metrics.epoch.v1",
    "digits.metrics.best.v1",
    "digits.metrics.artifact.v1",
    "digits.metrics.upload.v1",
    "digits.metrics.prune.v1",
    "digits.metrics.completed.v1",
]


class DigitsConfigV1(TypedDict):
    """Training configuration event published at job start."""

    type: Literal["digits.metrics.config.v1"]
    job_id: str
    user_id: int
    model_id: str
    total_epochs: int
    queue: str
    # Optional rich context
    cpu_cores: NotRequired[int]
    optimal_threads: NotRequired[int]
    memory_mb: NotRequired[int]
    optimal_workers: NotRequired[int]
    max_batch_size: NotRequired[int]
    device: NotRequired[str]
    # Optional augmentation/training hints
    batch_size: NotRequired[int]
    learning_rate: NotRequired[float]
    augment: NotRequired[bool]
    aug_rotate: NotRequired[float]
    aug_translate: NotRequired[float]
    noise_prob: NotRequired[float]
    dots_prob: NotRequired[float]


class DigitsBatchMetricsV1(TypedDict):
    """Batch-level metrics event published during training."""

    type: Literal["digits.metrics.batch.v1"]
    job_id: str
    user_id: int
    model_id: str
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    batch_loss: float
    batch_acc: float
    avg_loss: float
    samples_per_sec: float
    # Memory metrics (from cgroup-aware monitoring)
    main_rss_mb: int
    workers_rss_mb: int
    worker_count: int
    cgroup_usage_mb: int
    cgroup_limit_mb: int
    cgroup_pct: float
    anon_mb: int
    file_mb: int


class DigitsEpochMetricsV1(TypedDict):
    """Epoch-level metrics event published after each epoch."""

    type: Literal["digits.metrics.epoch.v1"]
    job_id: str
    user_id: int
    model_id: str
    epoch: int
    total_epochs: int
    train_loss: float
    val_acc: float
    time_s: float


class DigitsBestMetricsV1(TypedDict):
    """Best model metrics event published when a new best is found."""

    type: Literal["digits.metrics.best.v1"]
    job_id: str
    user_id: int
    model_id: str
    epoch: int
    val_acc: float


class DigitsArtifactV1(TypedDict):
    """Artifact saved event."""

    type: Literal["digits.metrics.artifact.v1"]
    job_id: str
    user_id: int
    model_id: str
    path: str


class DigitsUploadV1(TypedDict):
    """Upload status event."""

    type: Literal["digits.metrics.upload.v1"]
    job_id: str
    user_id: int
    model_id: str
    status: int
    model_bytes: int
    manifest_bytes: int
    file_id: str
    file_sha256: str


class DigitsPruneV1(TypedDict):
    """Prune/cleanup metrics event."""

    type: Literal["digits.metrics.prune.v1"]
    job_id: str
    user_id: int
    model_id: str
    deleted_count: int


class DigitsCompletedMetricsV1(TypedDict):
    """Training completion metrics event."""

    type: Literal["digits.metrics.completed.v1"]
    job_id: str
    user_id: int
    model_id: str
    val_acc: float


DigitsMetricsEventV1 = (
    DigitsConfigV1
    | DigitsBatchMetricsV1
    | DigitsEpochMetricsV1
    | DigitsBestMetricsV1
    | DigitsArtifactV1
    | DigitsUploadV1
    | DigitsPruneV1
    | DigitsCompletedMetricsV1
)


def encode_digits_metrics_event(event: DigitsMetricsEventV1) -> str:
    """Serialize a digits metrics event to a compact JSON string."""
    return dump_json_str(event)


# -----------------------------------------------------------------------------
# Factory functions for creating events
# -----------------------------------------------------------------------------


def _attach_optional_context(
    event: DigitsConfigV1,
    cpu_cores: int | None,
    optimal_threads: int | None,
    memory_mb: int | None,
    optimal_workers: int | None,
    max_batch_size: int | None,
    device: str | None,
) -> None:
    """Attach optional context fields to config event."""
    if cpu_cores is not None:
        event["cpu_cores"] = cpu_cores
    if optimal_threads is not None:
        event["optimal_threads"] = optimal_threads
    if memory_mb is not None:
        event["memory_mb"] = memory_mb
    if optimal_workers is not None:
        event["optimal_workers"] = optimal_workers
    if max_batch_size is not None:
        event["max_batch_size"] = max_batch_size
    if device is not None:
        event["device"] = device


def _attach_optional_augment(
    event: DigitsConfigV1,
    batch_size: int | None,
    learning_rate: float | None,
    augment: bool | None,
    aug_rotate: float | None,
    aug_translate: float | None,
    noise_prob: float | None,
    dots_prob: float | None,
) -> None:
    """Attach optional augmentation fields to config event."""
    if batch_size is not None:
        event["batch_size"] = batch_size
    if learning_rate is not None:
        event["learning_rate"] = learning_rate
    if augment is not None:
        event["augment"] = augment
    if aug_rotate is not None:
        event["aug_rotate"] = aug_rotate
    if aug_translate is not None:
        event["aug_translate"] = aug_translate
    if noise_prob is not None:
        event["noise_prob"] = noise_prob
    if dots_prob is not None:
        event["dots_prob"] = dots_prob


def make_config_event(
    *,
    job_id: str,
    user_id: int,
    model_id: str,
    total_epochs: int,
    queue: str,
    cpu_cores: int | None = None,
    optimal_threads: int | None = None,
    memory_mb: int | None = None,
    optimal_workers: int | None = None,
    max_batch_size: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    augment: bool | None = None,
    aug_rotate: float | None = None,
    aug_translate: float | None = None,
    noise_prob: float | None = None,
    dots_prob: float | None = None,
) -> DigitsConfigV1:
    """Create a digits training configuration event."""
    event: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "total_epochs": total_epochs,
        "queue": queue,
    }
    _attach_optional_context(
        event, cpu_cores, optimal_threads, memory_mb, optimal_workers, max_batch_size, device
    )
    _attach_optional_augment(
        event, batch_size, learning_rate, augment, aug_rotate, aug_translate, noise_prob, dots_prob
    )
    return event


def make_batch_metrics_event(
    *,
    job_id: str,
    user_id: int,
    model_id: str,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    batch_loss: float,
    batch_acc: float,
    avg_loss: float,
    samples_per_sec: float,
    main_rss_mb: int,
    workers_rss_mb: int,
    worker_count: int,
    cgroup_usage_mb: int,
    cgroup_limit_mb: int,
    cgroup_pct: float,
    anon_mb: int,
    file_mb: int,
) -> DigitsBatchMetricsV1:
    """Create a batch-level metrics event."""
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


def make_epoch_metrics_event(
    *,
    job_id: str,
    user_id: int,
    model_id: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_acc: float,
    time_s: float,
) -> DigitsEpochMetricsV1:
    """Create an epoch-level metrics event."""
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


def make_best_metrics_event(
    *,
    job_id: str,
    user_id: int,
    model_id: str,
    epoch: int,
    val_acc: float,
) -> DigitsBestMetricsV1:
    """Create a best model metrics event."""
    return {
        "type": "digits.metrics.best.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "epoch": epoch,
        "val_acc": val_acc,
    }


def make_artifact_event(
    *,
    job_id: str,
    user_id: int,
    model_id: str,
    path: str,
) -> DigitsArtifactV1:
    """Create an artifact saved event."""
    return {
        "type": "digits.metrics.artifact.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "path": path,
    }


def make_upload_event(
    *,
    job_id: str,
    user_id: int,
    model_id: str,
    status: int,
    model_bytes: int,
    manifest_bytes: int,
    file_id: str,
    file_sha256: str,
) -> DigitsUploadV1:
    """Create an upload status event."""
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


def make_prune_event(
    *,
    job_id: str,
    user_id: int,
    model_id: str,
    deleted_count: int,
) -> DigitsPruneV1:
    """Create a prune/cleanup metrics event."""
    return {
        "type": "digits.metrics.prune.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "deleted_count": deleted_count,
    }


def make_completed_metrics_event(
    *,
    job_id: str,
    user_id: int,
    model_id: str,
    val_acc: float,
) -> DigitsCompletedMetricsV1:
    """Create a training completion metrics event."""
    return {
        "type": "digits.metrics.completed.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "val_acc": val_acc,
    }


# -----------------------------------------------------------------------------
# Decoder functions
# -----------------------------------------------------------------------------

DecodedObj = dict[str, JSONValue]


def _decode_optional_int(decoded: DecodedObj, key: str) -> int | None:
    """Extract an optional int field from decoded dict."""
    val = decoded.get(key)
    return val if isinstance(val, int) else None


def _decode_optional_str(decoded: DecodedObj, key: str) -> str | None:
    """Extract an optional str field from decoded dict."""
    val = decoded.get(key)
    return val if isinstance(val, str) else None


def _decode_optional_float(decoded: DecodedObj, key: str) -> float | None:
    """Extract an optional float field from decoded dict (accepts int or float)."""
    val = decoded.get(key)
    return float(val) if isinstance(val, int | float) else None


def _decode_optional_bool(decoded: DecodedObj, key: str) -> bool | None:
    """Extract an optional bool field from decoded dict."""
    val = decoded.get(key)
    return val if isinstance(val, bool) else None


def _decode_config_context(event: DigitsConfigV1, decoded: DecodedObj) -> None:
    """Attach optional context fields to config event from decoded data."""
    cpu_cores = _decode_optional_int(decoded, "cpu_cores")
    if cpu_cores is not None:
        event["cpu_cores"] = cpu_cores
    optimal_threads = _decode_optional_int(decoded, "optimal_threads")
    if optimal_threads is not None:
        event["optimal_threads"] = optimal_threads
    memory_mb = _decode_optional_int(decoded, "memory_mb")
    if memory_mb is not None:
        event["memory_mb"] = memory_mb
    optimal_workers = _decode_optional_int(decoded, "optimal_workers")
    if optimal_workers is not None:
        event["optimal_workers"] = optimal_workers
    max_batch_size = _decode_optional_int(decoded, "max_batch_size")
    if max_batch_size is not None:
        event["max_batch_size"] = max_batch_size
    device = _decode_optional_str(decoded, "device")
    if device is not None:
        event["device"] = device


def _decode_config_augment(event: DigitsConfigV1, decoded: DecodedObj) -> None:
    """Attach optional augmentation fields to config event from decoded data."""
    batch_size = _decode_optional_int(decoded, "batch_size")
    if batch_size is not None:
        event["batch_size"] = batch_size
    learning_rate = _decode_optional_float(decoded, "learning_rate")
    if learning_rate is not None:
        event["learning_rate"] = learning_rate
    augment = _decode_optional_bool(decoded, "augment")
    if augment is not None:
        event["augment"] = augment
    aug_rotate = _decode_optional_float(decoded, "aug_rotate")
    if aug_rotate is not None:
        event["aug_rotate"] = aug_rotate
    aug_translate = _decode_optional_float(decoded, "aug_translate")
    if aug_translate is not None:
        event["aug_translate"] = aug_translate
    noise_prob = _decode_optional_float(decoded, "noise_prob")
    if noise_prob is not None:
        event["noise_prob"] = noise_prob
    dots_prob = _decode_optional_float(decoded, "dots_prob")
    if dots_prob is not None:
        event["dots_prob"] = dots_prob


def _decode_config_event(
    decoded: DecodedObj,
    job_id: str,
    user_id: int,
) -> DigitsConfigV1:
    model_id = decoded.get("model_id")
    total_epochs = decoded.get("total_epochs")
    queue = decoded.get("queue")
    if (
        not isinstance(model_id, str)
        or not isinstance(total_epochs, int)
        or not isinstance(queue, str)
    ):
        raise ValueError("config event requires model_id, total_epochs, queue")
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
    decoded: DecodedObj,
    job_id: str,
    user_id: int,
) -> DigitsBatchMetricsV1:
    model_id = decoded.get("model_id")
    epoch = decoded.get("epoch")
    total_epochs = decoded.get("total_epochs")
    batch = decoded.get("batch")
    total_batches = decoded.get("total_batches")
    batch_loss = decoded.get("batch_loss")
    batch_acc = decoded.get("batch_acc")
    avg_loss = decoded.get("avg_loss")
    samples_per_sec = decoded.get("samples_per_sec")
    main_rss_mb = decoded.get("main_rss_mb")
    workers_rss_mb = decoded.get("workers_rss_mb")
    worker_count = decoded.get("worker_count")
    cgroup_usage_mb = decoded.get("cgroup_usage_mb")
    cgroup_limit_mb = decoded.get("cgroup_limit_mb")
    cgroup_pct = decoded.get("cgroup_pct")
    anon_mb = decoded.get("anon_mb")
    file_mb = decoded.get("file_mb")
    if (
        not isinstance(model_id, str)
        or not isinstance(epoch, int)
        or not isinstance(total_epochs, int)
        or not isinstance(batch, int)
        or not isinstance(total_batches, int)
        or not isinstance(batch_loss, int | float)
        or not isinstance(batch_acc, int | float)
        or not isinstance(avg_loss, int | float)
        or not isinstance(samples_per_sec, int | float)
        or not isinstance(main_rss_mb, int)
        or not isinstance(workers_rss_mb, int)
        or not isinstance(worker_count, int)
        or not isinstance(cgroup_usage_mb, int)
        or not isinstance(cgroup_limit_mb, int)
        or not isinstance(cgroup_pct, int | float)
        or not isinstance(anon_mb, int)
        or not isinstance(file_mb, int)
    ):
        raise ValueError("batch metrics event missing required fields")
    return {
        "type": "digits.metrics.batch.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": batch,
        "total_batches": total_batches,
        "batch_loss": float(batch_loss),
        "batch_acc": float(batch_acc),
        "avg_loss": float(avg_loss),
        "samples_per_sec": float(samples_per_sec),
        "main_rss_mb": main_rss_mb,
        "workers_rss_mb": workers_rss_mb,
        "worker_count": worker_count,
        "cgroup_usage_mb": cgroup_usage_mb,
        "cgroup_limit_mb": cgroup_limit_mb,
        "cgroup_pct": float(cgroup_pct),
        "anon_mb": anon_mb,
        "file_mb": file_mb,
    }


def _decode_epoch_metrics_event(
    decoded: DecodedObj,
    job_id: str,
    user_id: int,
) -> DigitsEpochMetricsV1:
    model_id = decoded.get("model_id")
    epoch = decoded.get("epoch")
    total_epochs = decoded.get("total_epochs")
    train_loss = decoded.get("train_loss")
    val_acc = decoded.get("val_acc")
    time_s = decoded.get("time_s")
    if (
        not isinstance(model_id, str)
        or not isinstance(epoch, int)
        or not isinstance(total_epochs, int)
        or not isinstance(train_loss, int | float)
        or not isinstance(val_acc, int | float)
        or not isinstance(time_s, int | float)
    ):
        raise ValueError("epoch metrics event missing required fields")
    return {
        "type": "digits.metrics.epoch.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "train_loss": float(train_loss),
        "val_acc": float(val_acc),
        "time_s": float(time_s),
    }


def _decode_best_metrics_event(
    decoded: DecodedObj,
    job_id: str,
    user_id: int,
) -> DigitsBestMetricsV1:
    model_id = decoded.get("model_id")
    epoch = decoded.get("epoch")
    val_acc = decoded.get("val_acc")
    if (
        not isinstance(model_id, str)
        or not isinstance(epoch, int)
        or not isinstance(val_acc, int | float)
    ):
        raise ValueError("best metrics event missing required fields")
    return {
        "type": "digits.metrics.best.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "epoch": epoch,
        "val_acc": float(val_acc),
    }


def _decode_artifact_event(
    decoded: DecodedObj,
    job_id: str,
    user_id: int,
) -> DigitsArtifactV1:
    model_id = decoded.get("model_id")
    path = decoded.get("path")
    if not isinstance(model_id, str) or not isinstance(path, str):
        raise ValueError("artifact event missing required fields")
    return {
        "type": "digits.metrics.artifact.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "path": path,
    }


def _decode_upload_event(
    decoded: DecodedObj,
    job_id: str,
    user_id: int,
) -> DigitsUploadV1:
    model_id = decoded.get("model_id")
    status = decoded.get("status")
    model_bytes = decoded.get("model_bytes")
    manifest_bytes = decoded.get("manifest_bytes")
    file_id = decoded.get("file_id")
    file_sha256 = decoded.get("file_sha256")
    if (
        not isinstance(model_id, str)
        or not isinstance(status, int)
        or not isinstance(model_bytes, int)
        or not isinstance(manifest_bytes, int)
        or not isinstance(file_id, str)
        or not isinstance(file_sha256, str)
    ):
        raise ValueError("upload event missing required fields")
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
    decoded: DecodedObj,
    job_id: str,
    user_id: int,
) -> DigitsPruneV1:
    model_id = decoded.get("model_id")
    deleted_count = decoded.get("deleted_count")
    if not isinstance(model_id, str) or not isinstance(deleted_count, int):
        raise ValueError("prune event missing required fields")
    return {
        "type": "digits.metrics.prune.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "deleted_count": deleted_count,
    }


def _decode_completed_metrics_event(
    decoded: DecodedObj,
    job_id: str,
    user_id: int,
) -> DigitsCompletedMetricsV1:
    model_id = decoded.get("model_id")
    val_acc = decoded.get("val_acc")
    if not isinstance(model_id, str) or not isinstance(val_acc, int | float):
        raise ValueError("completed metrics event missing required fields")
    return {
        "type": "digits.metrics.completed.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_id": model_id,
        "val_acc": float(val_acc),
    }


_DECODERS: dict[
    str,
    Callable[[DecodedObj, str, int], DigitsMetricsEventV1],
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
        ValueError: if the payload is not a well-formed digits metrics event.
    """
    decoded_raw = load_json_str(payload)
    if not isinstance(decoded_raw, dict):
        raise ValueError("digits metrics event payload must be an object")
    decoded: DecodedObj = decoded_raw

    type_raw = decoded.get("type")
    if not isinstance(type_raw, str):
        raise ValueError("digits metrics event type must be a string")

    job_id = decoded.get("job_id")
    user_id = decoded.get("user_id")
    if not isinstance(job_id, str) or not isinstance(user_id, int):
        raise ValueError("job_id and user_id are required in digits metrics event")

    decoder = _DECODERS.get(type_raw)
    if decoder is None:
        raise ValueError("unknown digits metrics event type")
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
# Soft decoder for event routing (returns None instead of raising)
# -----------------------------------------------------------------------------


def _try_decode_metrics_event_impl(
    decoded: DecodedObj, type_raw: str, job_id: str, user_id: int
) -> DigitsMetricsEventV1 | None:
    """Attempt to decode a metrics event. Returns None if type is not recognized."""
    decoder = _DECODERS.get(type_raw)
    if decoder is None:
        return None
    # Decoder validates fields and raises ValueError if invalid
    return decoder(decoded, job_id, user_id)


def try_decode_digits_metrics_event(payload: str) -> DigitsMetricsEventV1 | None:
    """Try to decode a digits metrics event.

    Returns None if the payload is not a valid digits metrics event.
    Raises ValueError if the payload has the correct type but invalid fields.
    """
    decoded_raw = load_json_str(payload)
    if not isinstance(decoded_raw, dict):
        return None
    decoded: DecodedObj = decoded_raw

    type_raw = decoded.get("type")
    if not isinstance(type_raw, str):
        return None

    # Only handle digits.metrics.* events
    if not type_raw.startswith("digits.metrics."):
        return None

    job_id = decoded.get("job_id")
    user_id = decoded.get("user_id")
    if not isinstance(job_id, str) or not isinstance(user_id, int):
        return None

    return _try_decode_metrics_event_impl(decoded, type_raw, job_id, user_id)


# -----------------------------------------------------------------------------
# Combined event type for digits channel (job lifecycle + domain metrics)
# -----------------------------------------------------------------------------

# Combined event type for digits channel
DigitsEventV1 = JobEventV1 | DigitsMetricsEventV1

# Default channel for digits events
DEFAULT_DIGITS_EVENTS_CHANNEL: str = default_events_channel("digits")


def _try_decode_started(decoded: DecodedObj, job_id: str, user_id: int) -> JobStartedV1 | None:
    """Decode a started event."""
    queue = decoded.get("queue")
    if not isinstance(queue, str):
        return None
    return {
        "type": "digits.job.started.v1",
        "domain": "digits",
        "job_id": job_id,
        "user_id": user_id,
        "queue": queue,
    }


def _try_decode_completed(decoded: DecodedObj, job_id: str, user_id: int) -> JobCompletedV1 | None:
    """Decode a completed event."""
    result_id = decoded.get("result_id")
    result_bytes = decoded.get("result_bytes")
    if not isinstance(result_id, str) or not isinstance(result_bytes, int):
        return None
    return {
        "type": "digits.job.completed.v1",
        "domain": "digits",
        "job_id": job_id,
        "user_id": user_id,
        "result_id": result_id,
        "result_bytes": result_bytes,
    }


def _try_decode_failed(decoded: DecodedObj, job_id: str, user_id: int) -> JobFailedV1 | None:
    """Decode a failed event."""
    error_kind_raw = decoded.get("error_kind")
    message = decoded.get("message")
    if not isinstance(message, str):
        return None
    if error_kind_raw == "user":
        error_kind: Literal["user", "system"] = "user"
    elif error_kind_raw == "system":
        error_kind = "system"
    else:
        return None
    return {
        "type": "digits.job.failed.v1",
        "domain": "digits",
        "job_id": job_id,
        "user_id": user_id,
        "error_kind": error_kind,
        "message": message,
    }


def _try_decode_job_event_for_digits(decoded: DecodedObj, type_raw: str) -> JobEventV1 | None:
    """Attempt to decode a job lifecycle event for digits domain."""
    domain = decoded.get("domain")
    if domain != "digits":
        return None

    job_id = decoded.get("job_id")
    user_id = decoded.get("user_id")
    if not isinstance(job_id, str) or not isinstance(user_id, int):
        return None

    if type_raw == "digits.job.started.v1":
        return _try_decode_started(decoded, job_id, user_id)
    if type_raw == "digits.job.completed.v1":
        return _try_decode_completed(decoded, job_id, user_id)
    if type_raw == "digits.job.failed.v1":
        return _try_decode_failed(decoded, job_id, user_id)
    return None


def try_decode_digits_event(payload: str) -> DigitsEventV1 | None:
    """Try to decode any event from the digits channel.

    Handles both job lifecycle events (digits.job.*.v1) and
    metrics events (digits.metrics.*.v1).

    Returns None if the payload is not a valid digits event.
    Raises ValueError if the payload has the correct type but invalid fields.
    """
    decoded_raw = load_json_str(payload)
    if not isinstance(decoded_raw, dict):
        return None
    decoded: DecodedObj = decoded_raw

    type_raw = decoded.get("type")
    if not isinstance(type_raw, str):
        return None

    # Check if it's a job lifecycle event (digits.job.*.v1)
    if type_raw.startswith("digits.job."):
        return _try_decode_job_event_for_digits(decoded, type_raw)

    # Check if it's a metrics event (digits.metrics.*.v1)
    if type_raw.startswith("digits.metrics."):
        job_id = decoded.get("job_id")
        user_id = decoded.get("user_id")
        if not isinstance(job_id, str) or not isinstance(user_id, int):
            return None
        return _try_decode_metrics_event_impl(decoded, type_raw, job_id, user_id)

    return None


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
    "DigitsArtifactV1",
    "DigitsBatchMetricsV1",
    "DigitsBestMetricsV1",
    "DigitsCompletedMetricsV1",
    "DigitsConfigV1",
    "DigitsEpochMetricsV1",
    "DigitsEventV1",
    "DigitsMetricsEventType",
    "DigitsMetricsEventV1",
    "DigitsPruneV1",
    "DigitsUploadV1",
    "JobCompletedV1",
    "JobFailedV1",
    "JobStartedV1",
    "decode_digits_metrics_event",
    "encode_digits_metrics_event",
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
    "make_artifact_event",
    "make_batch_metrics_event",
    "make_best_metrics_event",
    "make_completed_metrics_event",
    "make_config_event",
    "make_epoch_metrics_event",
    "make_prune_event",
    "make_upload_event",
    "try_decode_digits_event",
    "try_decode_digits_metrics_event",
]
