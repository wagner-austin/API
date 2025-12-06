"""TypedDicts for wandb logging configuration and metrics.

These types define the schema for wandb run configuration and metrics logging.
Services can extend these TypedDicts for additional fields.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class WandbRunConfig(TypedDict):
    """Configuration logged at run start.

    Captures the essential training configuration for reproducibility.
    Services may add additional fields specific to their training setup.
    """

    run_id: str
    user_id: int
    model_family: str
    model_size: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    device: str


class WandbStepMetrics(TypedDict):
    """Per-step training metrics.

    Logged at configurable intervals during training to track progress.
    """

    train_loss: float
    train_ppl: float
    grad_norm: float
    samples_per_sec: float


class WandbEpochMetrics(TypedDict):
    """Epoch-end metrics.

    Logged at the end of each epoch with validation results.
    """

    epoch: int
    train_loss: float
    train_ppl: float
    val_loss: float
    val_ppl: float
    best_val_loss: float


class WandbFinalMetrics(TypedDict):
    """Training completion metrics.

    Logged once at the end of training with final evaluation results.
    """

    test_loss: float
    test_ppl: float
    early_stopped: bool


class WandbTableRow(TypedDict):
    """Single row for wandb table logging.

    Used for epoch summary tables with per-epoch metrics.
    """

    epoch: int
    train_loss: float
    train_ppl: float
    val_loss: float
    val_ppl: float


class WandbPublisherConfig(TypedDict):
    """Configuration for WandbPublisher initialization."""

    project: str
    run_name: str
    enabled: bool


class WandbInitResult(TypedDict):
    """Result of wandb initialization."""

    status: Literal["enabled", "disabled", "unavailable"]
    run_id: str | None


__all__ = [
    "WandbEpochMetrics",
    "WandbFinalMetrics",
    "WandbInitResult",
    "WandbPublisherConfig",
    "WandbRunConfig",
    "WandbStepMetrics",
    "WandbTableRow",
]
