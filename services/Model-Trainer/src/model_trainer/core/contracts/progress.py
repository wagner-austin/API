"""Training progress tracking contracts.

This module defines TypedDicts for tracking detailed training progress,
with encode/decode functions for Redis storage.

The progress structure captures the current phase of training, epoch/step
counts, loss metrics, and timing information for real-time monitoring.
"""

from __future__ import annotations

from typing import Final, Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    optional_float,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

TrainingPhase = Literal[
    "queued",
    "tokenization",
    "training",
    "validation",
    "test",
    "saving",
    "uploading",
    "completed",
    "failed",
    "cancelled",
]

_VALID_PHASES: Final[frozenset[str]] = frozenset(
    {
        "queued",
        "tokenization",
        "training",
        "validation",
        "test",
        "saving",
        "uploading",
        "completed",
        "failed",
        "cancelled",
    }
)


class TrainingProgress(TypedDict):
    """Detailed training progress for real-time monitoring.

    Attributes:
        run_id: Unique identifier for the training run.
        phase: Current phase of training.
        epoch: Current epoch number (0-indexed during training).
        total_epochs: Total number of epochs configured.
        step: Current step number within the epoch.
        total_steps: Total steps per epoch (0 if unknown).
        train_loss: Current training loss value.
        train_ppl: Current training perplexity.
        grad_norm: Current gradient norm value.
        samples_per_sec: Training throughput in samples per second.
        val_loss: Validation loss from last validation (None if not run yet).
        val_ppl: Validation perplexity from last validation (None if not run yet).
        updated_at: ISO 8601 timestamp of last update.
    """

    run_id: str
    phase: TrainingPhase
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    train_loss: float
    train_ppl: float
    grad_norm: float
    samples_per_sec: float
    val_loss: float | None
    val_ppl: float | None
    updated_at: str


def encode_training_progress(progress: TrainingProgress) -> JSONObject:
    """Encode TrainingProgress TypedDict to JSONObject for Redis storage.

    Args:
        progress: Training progress to encode.

    Returns:
        JSON-serializable dictionary with all progress fields.
    """
    return {
        "run_id": progress["run_id"],
        "phase": progress["phase"],
        "epoch": progress["epoch"],
        "total_epochs": progress["total_epochs"],
        "step": progress["step"],
        "total_steps": progress["total_steps"],
        "train_loss": progress["train_loss"],
        "train_ppl": progress["train_ppl"],
        "grad_norm": progress["grad_norm"],
        "samples_per_sec": progress["samples_per_sec"],
        "val_loss": progress["val_loss"],
        "val_ppl": progress["val_ppl"],
        "updated_at": progress["updated_at"],
    }


_PHASE_MAP: Final[dict[str, TrainingPhase]] = {
    "queued": "queued",
    "tokenization": "tokenization",
    "training": "training",
    "validation": "validation",
    "test": "test",
    "saving": "saving",
    "uploading": "uploading",
    "completed": "completed",
    "cancelled": "cancelled",
    "failed": "failed",
}


def _narrow_phase(raw: str) -> TrainingPhase:
    """Narrow phase string to TrainingPhase Literal with validation.

    Args:
        raw: Raw phase string.

    Returns:
        Narrowed TrainingPhase Literal type.

    Raises:
        JSONTypeError: If value is not a valid phase.
    """
    phase = _PHASE_MAP.get(raw)
    if phase is None:
        raise JSONTypeError(f"Field 'phase' must be one of {sorted(_VALID_PHASES)}, got '{raw}'")
    return phase


def decode_training_progress(obj: JSONObject) -> TrainingProgress:
    """Decode JSONObject to TrainingProgress with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TrainingProgress TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    run_id = require_str(obj, "run_id")
    phase = _narrow_phase(require_str(obj, "phase"))
    epoch = require_int(obj, "epoch")
    total_epochs = require_int(obj, "total_epochs")
    step = require_int(obj, "step")
    total_steps = require_int(obj, "total_steps")
    train_loss = require_float(obj, "train_loss")
    train_ppl = require_float(obj, "train_ppl")
    grad_norm = require_float(obj, "grad_norm")
    samples_per_sec = require_float(obj, "samples_per_sec")
    val_loss = optional_float(obj, "val_loss")
    val_ppl = optional_float(obj, "val_ppl")
    updated_at = require_str(obj, "updated_at")

    return {
        "run_id": run_id,
        "phase": phase,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "train_loss": train_loss,
        "train_ppl": train_ppl,
        "grad_norm": grad_norm,
        "samples_per_sec": samples_per_sec,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "updated_at": updated_at,
    }


def initial_progress(
    *,
    run_id: str,
    total_epochs: int,
    updated_at: str,
) -> TrainingProgress:
    """Create initial training progress in queued state.

    Args:
        run_id: Unique identifier for the training run.
        total_epochs: Total number of epochs configured.
        updated_at: ISO 8601 timestamp.

    Returns:
        TrainingProgress in queued state with zero metrics.
    """
    return {
        "run_id": run_id,
        "phase": "queued",
        "epoch": 0,
        "total_epochs": total_epochs,
        "step": 0,
        "total_steps": 0,
        "train_loss": 0.0,
        "train_ppl": 0.0,
        "grad_norm": 0.0,
        "samples_per_sec": 0.0,
        "val_loss": None,
        "val_ppl": None,
        "updated_at": updated_at,
    }


__all__ = [
    "TrainingPhase",
    "TrainingProgress",
    "decode_training_progress",
    "encode_training_progress",
    "initial_progress",
]
