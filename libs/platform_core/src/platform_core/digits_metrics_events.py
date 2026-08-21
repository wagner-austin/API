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

from typing import Literal, NotRequired, TypedDict

from .json_utils import (
    dump_json_str,
)

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


__all__ = [
    "DigitsArtifactV1",
    "DigitsBatchMetricsV1",
    "DigitsBestMetricsV1",
    "DigitsCompletedMetricsV1",
    "DigitsConfigV1",
    "DigitsEpochMetricsV1",
    "DigitsMetricsEventType",
    "DigitsMetricsEventV1",
    "DigitsPruneV1",
    "DigitsUploadV1",
    "encode_digits_metrics_event",
    "make_artifact_event",
    "make_batch_metrics_event",
    "make_best_metrics_event",
    "make_completed_metrics_event",
    "make_config_event",
    "make_epoch_metrics_event",
    "make_prune_event",
    "make_upload_event",
]
