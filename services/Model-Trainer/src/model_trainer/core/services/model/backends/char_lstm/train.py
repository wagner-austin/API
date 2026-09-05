"""CharLSTM training via BaseTrainer."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.determinism_record import DeterminismRecord
from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import (
    ModelTrainConfig,
    PreparedLMModel,
    ProgressCallback,
    TrainOutcome,
)
from model_trainer.core.services.training.base_trainer import BaseTrainer


def train_prepared_char_lstm(
    prepared: PreparedLMModel,
    cfg: ModelTrainConfig,
    settings: Settings,
    *,
    run_id: str,
    redis_hb: Callable[[float], None],
    cancelled: Callable[[], bool],
    resume: bool,
    progress: ProgressCallback | None = None,
    wandb_publisher: WandbPublisher | None = None,
    determinism: DeterminismRecord,
) -> TrainOutcome:
    """Train a prepared CharLSTM model.

    Args:
        prepared: Prepared model with tokenizer and config.
        cfg: Training configuration with model_family, model_size, etc.
        settings: Application settings.
        run_id: Unique identifier for this training run.
        redis_hb: Heartbeat callback (called with timestamp every 10 steps).
        cancelled: Callback to check if training was cancelled.
        resume: When True, continue the run from its persisted checkpoint.
        progress: Optional callback for progress updates
            (step, epoch, loss, grad_norm, samples_per_sec, val_loss, val_ppl).
        wandb_publisher: Optional wandb publisher for experiment tracking.

    Returns:
        TrainOutcome with loss, perplexity, steps, output directory, and
        cancellation status.
    """
    trainer = BaseTrainer(
        prepared,
        cfg,
        settings,
        run_id=run_id,
        redis_hb=redis_hb,
        cancelled=cancelled,
        resume=resume,
        progress=progress,
        service_name="char-lstm-train",
        wandb_publisher=wandb_publisher,
        determinism=determinism,
    )
    return trainer.train()
