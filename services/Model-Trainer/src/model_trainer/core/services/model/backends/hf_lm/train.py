"""HuggingFace LM training via BaseTrainer.

Uses hooks from _test_hooks for dependency injection.
Production sets hooks to real implementations at startup.
Tests set hooks to fakes for isolation.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel, TrainOutcome

from ._test_hooks import CreateTrainerFn, Hooks, ProgressCallback, TrainerProto


def train_prepared_hf_lm(
    prepared: PreparedLMModel,
    cfg: ModelTrainConfig,
    settings: Settings,
    *,
    run_id: str,
    redis_hb: Callable[[float], None],
    cancelled: Callable[[], bool],
    progress: ProgressCallback | None = None,
    wandb_publisher: WandbPublisher | None = None,
) -> TrainOutcome:
    """Train a HuggingFace LM model.

    Delegates to BaseTrainer (via hook) for the actual training loop.

    Args:
        prepared: Prepared model from prepare_hf_lm_with_handle.
        cfg: Training configuration.
        settings: Application settings.
        run_id: Unique identifier for this training run.
        redis_hb: Heartbeat callback (called with timestamp every 10 steps).
        cancelled: Callback to check if training was cancelled.
        progress: Optional callback for progress updates
            (step, epoch, loss, grad_norm, samples_per_sec, val_loss, val_ppl).
        wandb_publisher: Optional W&B publisher for metrics.

    Returns:
        TrainOutcome with final metrics and output directory.

    Raises:
        RuntimeError: If create_trainer hook is not initialized.
    """
    create_fn: CreateTrainerFn | None = Hooks.create_trainer
    if create_fn is None:
        raise RuntimeError("Hooks.create_trainer not initialized - call init_production_hooks()")

    trainer: TrainerProto = create_fn(
        prepared,
        cfg,
        settings,
        run_id=run_id,
        redis_hb=redis_hb,
        cancelled=cancelled,
        progress=progress,
        service_name="hf-lm-train",
        wandb_publisher=wandb_publisher,
    )
    return trainer.train()


__all__ = [
    "train_prepared_hf_lm",
]
