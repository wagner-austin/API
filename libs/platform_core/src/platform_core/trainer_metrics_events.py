from __future__ import annotations

from typing import Literal, NotRequired, TypedDict, TypeGuard

from .json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)

TrainerMetricsEventType = Literal[
    "trainer.metrics.config.v1",
    "trainer.metrics.progress.v1",
    "trainer.metrics.completed.v1",
]


class TrainerConfigV1(TypedDict):
    """Training configuration event published at job start."""

    type: Literal["trainer.metrics.config.v1"]
    job_id: str
    user_id: int
    model_family: str
    model_size: str
    total_epochs: int
    queue: str
    cpu_cores: NotRequired[int]
    memory_mb: NotRequired[int]
    optimal_threads: NotRequired[int]
    optimal_workers: NotRequired[int]
    batch_size: NotRequired[int]
    learning_rate: NotRequired[float]


class TrainerProgressMetricsV1(TypedDict):
    """Training progress metrics event published during training.

    Emitted per training step with core metrics. Validation metrics
    (val_loss, val_ppl) are only present at epoch boundaries.
    """

    type: Literal["trainer.metrics.progress.v1"]
    job_id: str
    user_id: int
    epoch: int
    total_epochs: int
    step: int
    train_loss: float
    train_ppl: float
    grad_norm: float
    samples_per_sec: float
    val_loss: NotRequired[float]
    val_ppl: NotRequired[float]


class TrainerCompletedMetricsV1(TypedDict):
    """Training completion metrics event published at job completion."""

    type: Literal["trainer.metrics.completed.v1"]
    job_id: str
    user_id: int
    test_loss: float
    test_ppl: float
    artifact_path: str


TrainerMetricsEventV1 = TrainerConfigV1 | TrainerProgressMetricsV1 | TrainerCompletedMetricsV1


def encode_trainer_metrics_event(event: TrainerMetricsEventV1) -> str:
    """Serialize a trainer metrics event to a compact JSON string."""
    return dump_json_str(event)


def make_config_event(
    *,
    job_id: str,
    user_id: int,
    model_family: str,
    model_size: str,
    total_epochs: int,
    queue: str,
    cpu_cores: int | None = None,
    memory_mb: int | None = None,
    optimal_threads: int | None = None,
    optimal_workers: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> TrainerConfigV1:
    """Create a training configuration event."""
    event: TrainerConfigV1 = {
        "type": "trainer.metrics.config.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_family": model_family,
        "model_size": model_size,
        "total_epochs": total_epochs,
        "queue": queue,
    }
    if cpu_cores is not None:
        event["cpu_cores"] = cpu_cores
    if memory_mb is not None:
        event["memory_mb"] = memory_mb
    if optimal_threads is not None:
        event["optimal_threads"] = optimal_threads
    if optimal_workers is not None:
        event["optimal_workers"] = optimal_workers
    if batch_size is not None:
        event["batch_size"] = batch_size
    if learning_rate is not None:
        event["learning_rate"] = learning_rate
    return event


def make_progress_metrics_event(
    *,
    job_id: str,
    user_id: int,
    epoch: int,
    total_epochs: int,
    step: int,
    train_loss: float,
    train_ppl: float,
    grad_norm: float,
    samples_per_sec: float,
    val_loss: float | None = None,
    val_ppl: float | None = None,
) -> TrainerProgressMetricsV1:
    """Create a training progress metrics event.

    Args:
        job_id: Unique identifier for the training job.
        user_id: User who initiated the training.
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs to train.
        step: Current training step.
        train_loss: Training loss for this step.
        train_ppl: Training perplexity for this step (exp(train_loss)).
        grad_norm: L2 norm of gradients after clipping.
        samples_per_sec: Training throughput.
        val_loss: Validation loss (only at epoch boundaries).
        val_ppl: Validation perplexity (only at epoch boundaries).

    Returns:
        TrainerProgressMetricsV1 event.
    """
    event: TrainerProgressMetricsV1 = {
        "type": "trainer.metrics.progress.v1",
        "job_id": job_id,
        "user_id": user_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "train_loss": train_loss,
        "train_ppl": train_ppl,
        "grad_norm": grad_norm,
        "samples_per_sec": samples_per_sec,
    }
    if val_loss is not None:
        event["val_loss"] = val_loss
    if val_ppl is not None:
        event["val_ppl"] = val_ppl
    return event


def make_completed_metrics_event(
    *,
    job_id: str,
    user_id: int,
    test_loss: float,
    test_ppl: float,
    artifact_path: str,
) -> TrainerCompletedMetricsV1:
    """Create a training completion metrics event."""
    return {
        "type": "trainer.metrics.completed.v1",
        "job_id": job_id,
        "user_id": user_id,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "artifact_path": artifact_path,
    }


def _decode_config_event(decoded: JSONObject, job_id: str, user_id: int) -> TrainerConfigV1:
    model_family = require_str(decoded, "model_family")
    model_size = require_str(decoded, "model_size")
    total_epochs = require_int(decoded, "total_epochs")
    queue = require_str(decoded, "queue")
    event: TrainerConfigV1 = {
        "type": "trainer.metrics.config.v1",
        "job_id": job_id,
        "user_id": user_id,
        "model_family": model_family,
        "model_size": model_size,
        "total_epochs": total_epochs,
        "queue": queue,
    }
    cpu_cores = decoded.get("cpu_cores")
    if isinstance(cpu_cores, int) and not isinstance(cpu_cores, bool):
        event["cpu_cores"] = cpu_cores
    memory_mb = decoded.get("memory_mb")
    if isinstance(memory_mb, int) and not isinstance(memory_mb, bool):
        event["memory_mb"] = memory_mb
    optimal_threads = decoded.get("optimal_threads")
    if isinstance(optimal_threads, int) and not isinstance(optimal_threads, bool):
        event["optimal_threads"] = optimal_threads
    optimal_workers = decoded.get("optimal_workers")
    if isinstance(optimal_workers, int) and not isinstance(optimal_workers, bool):
        event["optimal_workers"] = optimal_workers
    batch_size = decoded.get("batch_size")
    if isinstance(batch_size, int) and not isinstance(batch_size, bool):
        event["batch_size"] = batch_size
    learning_rate = decoded.get("learning_rate")
    if isinstance(learning_rate, int | float) and not isinstance(learning_rate, bool):
        event["learning_rate"] = float(learning_rate)
    return event


def _decode_progress_metrics_event(
    decoded: JSONObject, job_id: str, user_id: int
) -> TrainerProgressMetricsV1:
    epoch = require_int(decoded, "epoch")
    total_epochs = require_int(decoded, "total_epochs")
    step = require_int(decoded, "step")
    train_loss = require_float(decoded, "train_loss")
    train_ppl = require_float(decoded, "train_ppl")
    grad_norm = require_float(decoded, "grad_norm")
    samples_per_sec = require_float(decoded, "samples_per_sec")
    event: TrainerProgressMetricsV1 = {
        "type": "trainer.metrics.progress.v1",
        "job_id": job_id,
        "user_id": user_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "train_loss": train_loss,
        "train_ppl": train_ppl,
        "grad_norm": grad_norm,
        "samples_per_sec": samples_per_sec,
    }
    val_loss = decoded.get("val_loss")
    if isinstance(val_loss, int | float) and not isinstance(val_loss, bool):
        event["val_loss"] = float(val_loss)
    val_ppl = decoded.get("val_ppl")
    if isinstance(val_ppl, int | float) and not isinstance(val_ppl, bool):
        event["val_ppl"] = float(val_ppl)
    return event


def _decode_completed_metrics_event(
    decoded: JSONObject, job_id: str, user_id: int
) -> TrainerCompletedMetricsV1:
    test_loss = require_float(decoded, "test_loss")
    test_ppl = require_float(decoded, "test_ppl")
    artifact_path = require_str(decoded, "artifact_path")
    return {
        "type": "trainer.metrics.completed.v1",
        "job_id": job_id,
        "user_id": user_id,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "artifact_path": artifact_path,
    }


def decode_trainer_metrics_event(payload: str) -> TrainerMetricsEventV1:
    """Parse and validate a serialized trainer metrics event.

    Raises:
        JSONTypeError: if the payload is not a well-formed trainer metrics event.
    """
    decoded = narrow_json_to_dict(load_json_str(payload))

    type_raw = require_str(decoded, "type")
    job_id = require_str(decoded, "job_id")
    user_id = require_int(decoded, "user_id")

    if type_raw == "trainer.metrics.config.v1":
        return _decode_config_event(decoded, job_id, user_id)
    if type_raw == "trainer.metrics.progress.v1":
        return _decode_progress_metrics_event(decoded, job_id, user_id)
    if type_raw == "trainer.metrics.completed.v1":
        return _decode_completed_metrics_event(decoded, job_id, user_id)
    raise JSONTypeError(f"Unknown trainer metrics event type: '{type_raw}'")


def is_config(ev: TrainerMetricsEventV1) -> TypeGuard[TrainerConfigV1]:
    """Check if the event is a config event."""
    return ev.get("type") == "trainer.metrics.config.v1"


def is_progress_metrics(ev: TrainerMetricsEventV1) -> TypeGuard[TrainerProgressMetricsV1]:
    """Check if the event is a progress metrics event."""
    return ev.get("type") == "trainer.metrics.progress.v1"


def is_completed_metrics(ev: TrainerMetricsEventV1) -> TypeGuard[TrainerCompletedMetricsV1]:
    """Check if the event is a completed metrics event."""
    return ev.get("type") == "trainer.metrics.completed.v1"


__all__ = [
    "TrainerCompletedMetricsV1",
    "TrainerConfigV1",
    "TrainerMetricsEventType",
    "TrainerMetricsEventV1",
    "TrainerProgressMetricsV1",
    "decode_trainer_metrics_event",
    "encode_trainer_metrics_event",
    "is_completed_metrics",
    "is_config",
    "is_progress_metrics",
    "make_completed_metrics_event",
    "make_config_event",
    "make_progress_metrics_event",
]
