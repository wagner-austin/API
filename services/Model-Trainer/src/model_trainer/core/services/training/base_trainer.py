"""Base trainer for language models: the public entry point.

Assembled from the linear chain core -> observability -> checkpoints ->
loop; this module adds :meth:`BaseTrainer.train`.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import torch
from platform_core.logging import get_logger

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.model import (
    EarlyStoppingState,
    TrainOutcome,
)
from model_trainer.core.services.training.base_trainer_loop import (
    _TrainerLoop,
)
from model_trainer.core.services.training.checkpoint import (
    TrainingCheckpoint,
    delete_training_checkpoint,
    load_training_checkpoint,
)

_logger = get_logger(__name__)


class BaseTrainer(_TrainerLoop):
    """Base trainer for language models (see the chain modules)."""

    def train(self) -> TrainOutcome:
        """Execute the training loop with early stopping and validation.

        Returns:
            TrainOutcome with loss, perplexity, steps, output directory,
            cancellation status, test metrics, and early stopping info.
        """
        torch.manual_seed(self._cfg["seed"])
        random.seed(self._cfg["seed"])

        # Initialize training metrics tracking
        self._training_start_time = _test_hooks.time_monotonic()
        self._training_start_iso = _test_hooks.datetime_utcnow_iso()
        self._total_samples_processed = 0
        self._total_tokens_processed = 0

        # 1. Setup device (NEW - GPU support)
        self._device = self._setup_device()

        # Reset GPU peak memory stats before training
        _test_hooks.gpu_reset_peak_memory_stats()

        # 2. Apply LR cap if fine-tuning (NEW)
        effective_lr = self._apply_lr_cap()

        # 3. Build data loaders (UPDATED - now builds train/val/test)
        train_loader, self._val_loader, self._test_loader = self._build_all_loaders()

        # Log config to wandb at start of training
        self._log_wandb_config()

        model = self._prepared.model
        model.train()
        model.to(str(self._device))

        # Freeze embeddings if configured
        if self._cfg["freeze_embed"]:
            _test_hooks.freeze_embeddings(model)

        # 4. Initialize early stopping (NEW)
        self._es_state = EarlyStoppingState(
            best_val_loss=float("inf"),
            epochs_no_improve=0,
        )
        self._best_checkpoint_path = None

        # 4b. Apply the persisted checkpoint when resuming
        restored: TrainingCheckpoint | None = None
        start_epoch = 0
        start_step = 0
        initial_last_loss = 0.0
        if self._resume:
            restored = load_training_checkpoint(self._settings, self._run_id)
            self._require_matching_config(restored.meta)
            self._apply_checkpoint(restored)
            start_epoch = restored.meta["epochs_completed"]
            start_step = restored.meta["global_step"]
            initial_last_loss = restored.meta["last_loss"]

        # 5. Run training loop (UPDATED - returns more info)
        last_loss, step, was_cancelled, early_stopped = self._run_training_loop(
            model,
            train_loader,
            effective_lr,
            start_epoch=start_epoch,
            start_step=start_step,
            initial_last_loss=initial_last_loss,
            restored=restored,
        )

        out_dir = str(_test_hooks.model_dir(self._settings, self._run_id))
        os.makedirs(out_dir, exist_ok=True)

        # 6. Save checkpoint if not cancelled and no best was saved
        if not was_cancelled and self._best_checkpoint_path is None:
            _logger.info(
                "Saving final model checkpoint",
                extra={
                    "category": "training",
                    "event": "model_save_started",
                    "out_dir": out_dir,
                    "run_id": self._run_id,
                },
            )
            self._prepared.model.save_pretrained(out_dir)
            _logger.info(
                "Final model checkpoint saved",
                extra={
                    "category": "training",
                    "event": "model_save_completed",
                    "out_dir": out_dir,
                    "run_id": self._run_id,
                },
            )

        # 6b. Score what ships, not what happens to be resident. With a
        # holdout, out_dir holds the best epoch while the live model holds
        # the last; without one, this is a no-op.
        self._restore_best_checkpoint()

        # 7. Run test evaluation (NEW)
        test_loss: float | None = None
        test_ppl: float | None = None
        if not was_cancelled and self._test_loader is not None:
            test_metrics = self._run_evaluation(self._test_loader, eval_type="test")
            test_loss = test_metrics["val_loss"]
            test_ppl = test_metrics["val_ppl"]

        # Get best val loss (may be inf if no validation was done)
        best_val_loss: float | None = None
        if self._es_state["best_val_loss"] < float("inf"):
            best_val_loss = self._es_state["best_val_loss"]

        # 8. Compute training metrics for manifest. Duration accumulates
        # across executions: prior executions' time comes from the
        # checkpoint, this execution's from the monotonic clock.
        training_end_time = _test_hooks.time_monotonic()
        training_end_iso = _test_hooks.datetime_utcnow_iso()
        training_duration_sec = (
            training_end_time - self._training_start_time
        ) + self._elapsed_before

        # Get GPU peak memory (None if CPU training)
        peak_gpu_memory_bytes = _test_hooks.gpu_max_memory_allocated()
        peak_gpu_memory_mb: float | None = None
        if self._cfg["device"] == "cuda" and peak_gpu_memory_bytes > 0:
            peak_gpu_memory_mb = peak_gpu_memory_bytes / (1024 * 1024)

        # Compute average throughput
        avg_samples_per_sec = self._total_samples_processed / max(training_duration_sec, 0.001)

        # Get model info
        param_count = _test_hooks.count_model_parameters(self._prepared.model)
        model_size_bytes = _test_hooks.get_directory_size_bytes(Path(out_dir))
        model_size_mb = model_size_bytes / (1024 * 1024)
        vocab_size = self._prepared.tok_for_dataset.get_vocab_size()

        self._write_manifest(
            out_dir=out_dir,
            steps=step,
            last_loss=last_loss,
            test_loss=test_loss,
            test_ppl=test_ppl,
            best_val_loss=best_val_loss,
            early_stopped=early_stopped,
            training_duration_sec=training_duration_sec,
            started_at=self._training_start_iso,
            completed_at=training_end_iso,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            avg_samples_per_sec=avg_samples_per_sec,
            total_tokens_processed=self._total_tokens_processed,
            param_count=param_count,
            model_size_mb=model_size_mb,
            vocab_size=vocab_size,
        )

        # Log final metrics and epoch table to wandb
        self._log_wandb_final(
            test_loss=test_loss,
            test_ppl=test_ppl,
            early_stopped=early_stopped,
        )
        self._log_wandb_epoch_table()
        self._finish_wandb()

        # A finished run's final model and manifest supersede its resume
        # state; a cancelled run keeps the checkpoint so it stays
        # explicitly resumable.
        if not was_cancelled:
            _ = delete_training_checkpoint(self._settings, self._run_id)

        ppl = float(math.exp(last_loss)) if last_loss < 20 else float("inf")
        return TrainOutcome(
            loss=last_loss,
            perplexity=ppl,
            steps=step,
            out_dir=out_dir,
            cancelled=was_cancelled,
            test_loss=test_loss,
            test_perplexity=test_ppl,
            best_val_loss=best_val_loss,
            early_stopped=early_stopped,
        )
