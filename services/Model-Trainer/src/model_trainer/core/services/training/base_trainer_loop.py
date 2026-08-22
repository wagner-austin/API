"""BaseTrainer loop: evaluation, epochs, the training loop."""

from __future__ import annotations

import math
from typing import Literal

import torch
from platform_core.logging import get_logger

from model_trainer.core.contracts.model import (
    ValidationMetrics,
)
from model_trainer.core.services.training.base_trainer_checkpoints import (
    _TrainerCheckpoints,
)
from model_trainer.core.services.training.base_trainer_core import (
    _get_optimizer_for_config,
)
from model_trainer.core.services.training.checkpoint import (
    TrainingCheckpoint,
)
from model_trainer.core.services.training.dataloader import DataLoader
from model_trainer.core.services.training.trainer_grad_utils import (
    _clip_grad_norm_with_return,
    _create_grad_scaler,
    _get_autocast_context,
    _GradScalerProto,
)
from model_trainer.core.types import (
    LMModelProto,
    OptimizerProto,
)

_logger = get_logger(__name__)


class _TrainerLoop(_TrainerCheckpoints):
    """Evaluation and the epoch/training loops."""

    def _run_evaluation(
        self,
        loader: DataLoader,
        *,
        eval_type: Literal["validation", "test"] = "validation",
    ) -> ValidationMetrics:
        """Run evaluation on given loader with progress logging.

        Uses the same precision as training for consistent metrics.

        Args:
            loader: DataLoader to evaluate on.
            eval_type: Type of evaluation for logging ("validation" or "test").

        Returns:
            ValidationMetrics with loss and perplexity.
        """
        total_batches = len(loader)
        _logger.info(
            "%s started",
            eval_type.capitalize(),
            extra={
                "category": "training",
                "event": f"{eval_type}_started",
                "total_batches": total_batches,
                "run_id": self._run_id,
            },
        )

        model = self._prepared.model
        model.eval()

        total_loss = 0.0
        num_batches = 0
        device_str = str(self._device)

        # Use same precision as training for consistent metrics
        precision = self._cfg["precision"]
        autocast_ctx = _get_autocast_context(precision, self._device)

        # Log progress every 10% or at least every 100 batches
        log_interval = max(1, min(100, total_batches // 10))

        with torch.no_grad():
            for batch in loader:
                # Check cancellation during evaluation
                if self._cancelled():
                    _logger.info(
                        "%s cancelled at batch %d/%d",
                        eval_type.capitalize(),
                        num_batches,
                        total_batches,
                        extra={
                            "category": "training",
                            "event": f"{eval_type}_cancelled",
                            "batch": num_batches,
                            "total_batches": total_batches,
                        },
                    )
                    model.train()
                    # Return partial results
                    avg_loss = total_loss / max(1, num_batches)
                    avg_ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")
                    return ValidationMetrics(val_loss=avg_loss, val_ppl=avg_ppl)

                inputs = batch[0].to(device_str)
                labels = batch[1].to(device_str)
                with autocast_ctx:
                    outputs = model.forward(input_ids=inputs, labels=labels)
                total_loss += float(outputs.loss.item())
                num_batches += 1

                # Log progress periodically
                if num_batches % log_interval == 0:
                    progress_pct = int((num_batches * 100) / total_batches)
                    running_avg_loss = total_loss / num_batches
                    _logger.info(
                        "%s progress batch=%d/%d (%.0f%%) running_loss=%.4f",
                        eval_type.capitalize(),
                        num_batches,
                        total_batches,
                        progress_pct,
                        running_avg_loss,
                        extra={
                            "category": "training",
                            "event": f"{eval_type}_progress",
                            "batch": num_batches,
                            "total_batches": total_batches,
                            "progress_pct": progress_pct,
                            "running_loss": running_avg_loss,
                        },
                    )

        model.train()
        avg_loss = total_loss / max(1, num_batches)
        avg_ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")

        _logger.info(
            "%s completed batches=%d loss=%.4f ppl=%.2f",
            eval_type.capitalize(),
            num_batches,
            avg_loss,
            avg_ppl,
            extra={
                "category": "training",
                "event": f"{eval_type}_completed",
                "batches": num_batches,
                "loss": avg_loss,
                "ppl": avg_ppl,
            },
        )

        return ValidationMetrics(val_loss=avg_loss, val_ppl=avg_ppl)

    def _run_training_loop(
        self,
        model: LMModelProto,
        dataloader: DataLoader,
        effective_lr: float,
        *,
        start_epoch: int,
        start_step: int,
        initial_last_loss: float,
        restored: TrainingCheckpoint | None,
    ) -> tuple[float, int, bool, bool]:
        """Run the main training loop with early stopping.

        Args:
            model: The language model to train.
            dataloader: Training data loader.
            effective_lr: Learning rate (potentially capped for fine-tuning).
            start_epoch: Epoch index to start from; non-zero on resume.
            start_step: Global step count already taken; non-zero on resume.
            initial_last_loss: Loss carried from the checkpoint, reported
                unchanged when the loop body never runs.
            restored: Checkpoint whose optimizer state to apply, or None
                when training from scratch.

        Returns:
            Tuple of (final_loss, total_steps, was_cancelled, early_stopped).
        """
        optimizer_cls = _get_optimizer_for_config(self._cfg["optimizer"])
        optim = optimizer_cls(model.parameters(), lr=effective_lr)
        if restored is not None:
            optim.load_state_dict(restored.optimizer_state)
        step = start_step
        last_loss = initial_last_loss
        was_cancelled = False
        early_stopped = False
        device_str = str(self._device)

        total_epochs = self._cfg["num_epochs"]
        batches_per_epoch = len(dataloader)

        for epoch in range(start_epoch, total_epochs):
            epoch_step_start = step
            _logger.info(
                "Epoch %d/%d started batches=%d",
                epoch + 1,
                total_epochs,
                batches_per_epoch,
                extra={
                    "category": "training",
                    "event": "epoch_started",
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "batches": batches_per_epoch,
                    "run_id": self._run_id,
                },
            )
            last_loss, step, was_cancelled, avg_grad_norm = self._train_one_epoch(
                model=model,
                dataloader=dataloader,
                optim=optim,
                epoch=epoch,
                device=device_str,
                start_step=step,
            )
            epoch_steps = step - epoch_step_start
            _logger.info(
                "Epoch %d/%d completed steps=%d loss=%.4f",
                epoch + 1,
                total_epochs,
                epoch_steps,
                last_loss,
                extra={
                    "category": "training",
                    "event": "epoch_completed",
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "steps": epoch_steps,
                    "loss": last_loss,
                    "run_id": self._run_id,
                },
            )
            if was_cancelled:
                break

            # Report progress for empty epochs
            if self._progress is not None and step == epoch_step_start:
                ppl = float(math.exp(last_loss)) if last_loss < 20 else float("inf")
                self._progress(step, epoch, last_loss, ppl, 0.0, 0.0, None, None)

            # Run validation after each epoch
            if self._val_loader is not None:
                val_metrics = self._run_evaluation(self._val_loader, eval_type="validation")

                # Calculate train_ppl for progress and wandb logging
                train_ppl = float(math.exp(last_loss)) if last_loss < 20 else float("inf")

                # Emit progress with validation metrics at epoch boundary
                if self._progress is not None:
                    self._progress(
                        step,
                        epoch,
                        last_loss,
                        train_ppl,
                        avg_grad_norm,
                        0.0,  # samples_per_sec not meaningful at epoch end
                        val_metrics["val_loss"],
                        val_metrics["val_ppl"],
                    )

                # Log epoch metrics to wandb
                self._log_wandb_epoch(
                    epoch=epoch,
                    train_loss=last_loss,
                    train_ppl=train_ppl,
                    val_loss=val_metrics["val_loss"],
                    val_ppl=val_metrics["val_ppl"],
                    best_val_loss=self._es_state["best_val_loss"],
                    epochs_no_improve=self._es_state["epochs_no_improve"],
                )

                # Track epoch summary for final table
                self._epoch_summaries.append(
                    (epoch, last_loss, train_ppl, val_metrics["val_loss"], val_metrics["val_ppl"])
                )

                # Check for improvement (NEW)
                if val_metrics["val_loss"] < self._es_state["best_val_loss"]:
                    self._es_state["best_val_loss"] = val_metrics["val_loss"]
                    self._es_state["epochs_no_improve"] = 0
                    # Save best checkpoint (NEW)
                    self._save_best_checkpoint()
                else:
                    self._es_state["epochs_no_improve"] += 1

                # Check early stopping (patience=0 disables early stopping)
                patience = self._cfg["early_stopping_patience"]
                if patience > 0 and self._es_state["epochs_no_improve"] >= patience:
                    early_stopped = True
                    _logger.info(
                        "Early stopping triggered",
                        extra={
                            "category": "training",
                            "event": "early_stopping",
                            "epochs_no_improve": self._es_state["epochs_no_improve"],
                            "patience": patience,
                        },
                    )

            # An early-stopped run completes in this execution and needs
            # no resume state; every other completed epoch publishes the
            # rolling checkpoint so an interruption costs at most one
            # epoch.
            if early_stopped:
                break
            self._save_epoch_checkpoint(
                model=model,
                optim=optim,
                epochs_completed=epoch + 1,
                global_step=step,
                last_loss=last_loss,
            )

        return last_loss, step, was_cancelled, early_stopped

    def _train_one_epoch(
        self,
        *,
        model: LMModelProto,
        dataloader: DataLoader,
        optim: OptimizerProto,
        epoch: int,
        device: str,
        start_step: int,
    ) -> tuple[float, int, bool, float]:
        """Train for one epoch with gradient norm and throughput tracking.

        Supports mixed precision training:
        - fp32: Standard training without autocast
        - fp16: Uses autocast + GradScaler for stability
        - bf16: Uses autocast without scaler (bf16 is more numerically stable)

        Args:
            model: The language model.
            dataloader: Training data loader.
            optim: Optimizer instance.
            epoch: Current epoch number.
            device: Device to train on.
            start_step: Step number at start of epoch.

        Returns:
            Tuple of (last_loss, current_step, was_cancelled, avg_grad_norm).
        """
        import time as _time

        step = start_step
        last_loss = 0.0
        total_grad_norm = 0.0
        grad_norm_count = 0
        batch_size = self._cfg["batch_size"]

        # Precision setup
        precision = self._cfg["precision"]
        use_fp16_scaler = precision == "fp16" and self._device.type == "cuda"
        autocast_ctx = _get_autocast_context(precision, self._device)
        scaler: _GradScalerProto | None = _create_grad_scaler() if use_fp16_scaler else None

        # Throughput tracking
        samples_processed = 0
        epoch_start_time = _time.time()

        for batch in dataloader:
            if self._cancelled():
                avg_grad_norm = total_grad_norm / max(1, grad_norm_count)
                return last_loss, step, True, avg_grad_norm

            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass with autocast (no-op for fp32)
            with autocast_ctx:
                outputs = model.forward(input_ids=inputs, labels=labels)
                loss_t = outputs.loss

            last_loss = float(loss_t.item())
            optim.zero_grad(set_to_none=True)

            # Backward pass: scaled for fp16, standard for fp32/bf16
            if scaler is not None:
                scaled_loss = scaler.scale(loss_t)
                torch.autograd.backward([scaled_loss])
                scaler.unscale_(optim)
            else:
                torch.autograd.backward([loss_t])

            # Capture gradient norm BEFORE clipping
            grad_norm = _clip_grad_norm_with_return(
                model.parameters(),
                max_norm=self._cfg["gradient_clipping"],
            )
            total_grad_norm += grad_norm
            grad_norm_count += 1

            # Optimizer step: through scaler for fp16, standard otherwise
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()

            step += 1
            samples_processed += batch_size

            # Track tokens processed (batch_size * sequence_length)
            tokens_in_batch = inputs.numel()
            self._total_samples_processed += batch_size
            self._total_tokens_processed += tokens_in_batch

            # Compute current throughput
            elapsed = _time.time() - epoch_start_time
            samples_per_sec = samples_processed / max(elapsed, 0.001)

            # Compute train ppl once for both progress and wandb
            train_ppl = float(math.exp(last_loss)) if last_loss < 20 else float("inf")

            if self._progress is not None:
                # Per-step progress: no val metrics (those come at epoch end)
                self._progress(
                    step,
                    epoch,
                    last_loss,
                    train_ppl,
                    grad_norm,
                    samples_per_sec,
                    None,
                    None,
                )

            # Log step metrics to wandb
            self._log_wandb_step(
                step=step,
                epoch=epoch,
                train_loss=last_loss,
                train_ppl=train_ppl,
                grad_norm=grad_norm,
                samples_per_sec=samples_per_sec,
            )

            if step % 10 == 0:
                self._redis_hb(_time.time())

        avg_grad_norm = total_grad_norm / max(1, grad_norm_count)
        return last_loss, step, False, avg_grad_norm
