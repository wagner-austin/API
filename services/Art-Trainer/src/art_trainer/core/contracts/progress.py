"""Training progress contract with encode/decode functions.

This module defines the ArtTrainingProgress TypedDict for tracking
training state and provides encode/decode functions for serialization.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    optional_float,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

ArtTrainingPhase = Literal[
    "queued",
    "preparing",
    "training",
    "saving",
    "uploading",
    "completed",
    "failed",
    "cancelled",
]


class ArtTrainingProgress(TypedDict, total=True):
    """Training progress state for art model training.

    Attributes:
        job_id: Unique job identifier.
        phase: Current training phase.
        step: Current training step (0-indexed).
        total_steps: Total number of steps.
        loss: Current loss value (None if not available).
        learning_rate: Current learning rate.
        updated_at: ISO 8601 timestamp of last update.
    """

    job_id: str
    phase: ArtTrainingPhase
    step: int
    total_steps: int
    loss: float | None
    learning_rate: float
    updated_at: str


def _narrow_phase(raw: str) -> ArtTrainingPhase:
    """Narrow phase string to Literal type with validation.

    Args:
        raw: Raw phase string.

    Returns:
        Narrowed Literal type.

    Raises:
        JSONTypeError: If value is not a valid phase.
    """
    if raw == "queued":
        return "queued"
    if raw == "preparing":
        return "preparing"
    if raw == "training":
        return "training"
    if raw == "saving":
        return "saving"
    if raw == "uploading":
        return "uploading"
    if raw == "completed":
        return "completed"
    if raw == "failed":
        return "failed"
    if raw == "cancelled":
        return "cancelled"
    raise JSONTypeError(f"Field 'phase' must be a valid training phase, got '{raw}'")


def encode_art_training_progress(progress: ArtTrainingProgress) -> JSONObject:
    """Encode ArtTrainingProgress to JSONObject for serialization.

    Args:
        progress: Training progress to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "job_id": progress["job_id"],
        "phase": progress["phase"],
        "step": progress["step"],
        "total_steps": progress["total_steps"],
        "loss": progress["loss"],
        "learning_rate": progress["learning_rate"],
        "updated_at": progress["updated_at"],
    }


def decode_art_training_progress(obj: JSONObject) -> ArtTrainingProgress:
    """Decode JSONObject to ArtTrainingProgress with full validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated ArtTrainingProgress TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    job_id = require_str(obj, "job_id")
    phase = _narrow_phase(require_str(obj, "phase"))
    step = require_int(obj, "step")
    total_steps = require_int(obj, "total_steps")
    loss = optional_float(obj, "loss")
    learning_rate = require_float(obj, "learning_rate")
    updated_at = require_str(obj, "updated_at")

    return {
        "job_id": job_id,
        "phase": phase,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "learning_rate": learning_rate,
        "updated_at": updated_at,
    }


__all__ = [
    "ArtTrainingPhase",
    "ArtTrainingProgress",
    "decode_art_training_progress",
    "encode_art_training_progress",
]
