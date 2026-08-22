"""BaseTrainer checkpointing: save, verify, restore."""

from __future__ import annotations

import os
from pathlib import Path

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.logging import get_logger

from model_trainer.core import _test_hooks
from model_trainer.core.contracts.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    EpochSummaryRecord,
    TrainingCheckpointMeta,
    model_train_config_mismatches,
)
from model_trainer.core.contracts.model import (
    EarlyStoppingState,
)
from model_trainer.core.services.training.base_trainer_observability import (
    _TrainerObservability,
)
from model_trainer.core.services.training.checkpoint import (
    TrainingCheckpoint,
    capture_rng_states,
    restore_rng_states,
    save_training_checkpoint,
)
from model_trainer.core.types import (
    LMModelProto,
    OptimizerProto,
)

_logger = get_logger(__name__)


class _TrainerCheckpoints(_TrainerObservability):
    """Checkpoint lifecycle over the observable trainer."""

    def _save_best_checkpoint(self) -> None:
        """Save current model as best checkpoint."""
        out_dir = str(_test_hooks.model_dir(self._settings, self._run_id))
        os.makedirs(out_dir, exist_ok=True)
        self._prepared.model.save_pretrained(out_dir)
        self._best_checkpoint_path = Path(out_dir)
        _logger.info(
            "Saved best checkpoint",
            extra={
                "category": "training",
                "event": "checkpoint_saved",
                "path": out_dir,
            },
        )

    def _require_matching_config(self, meta: TrainingCheckpointMeta) -> None:
        """Refuse to resume under a config that differs from the checkpoint's.

        Args:
            meta: Metadata loaded from the run's checkpoint.

        Raises:
            AppError: With ``CHECKPOINT_CONFIG_MISMATCH`` naming every
                differing field. A resume must reproduce the interrupted
                run exactly; a changed config describes a different
                experiment and needs a fresh run.
        """
        mismatches = model_train_config_mismatches(meta["config"], self._cfg)
        if mismatches:
            raise AppError(
                ModelTrainerErrorCode.CHECKPOINT_CONFIG_MISMATCH,
                (
                    f"run '{self._run_id}' cannot resume: the submitted config "
                    f"differs from the checkpoint's on: {', '.join(mismatches)}"
                ),
                model_trainer_status_for(ModelTrainerErrorCode.CHECKPOINT_CONFIG_MISMATCH),
            )

    def _apply_checkpoint(self, restored: TrainingCheckpoint) -> None:
        """Restore trainer state from a loaded checkpoint.

        Loads model weights, early-stopping state, progress counters,
        epoch summaries, accumulated timing and every RNG state, so the
        next epoch proceeds exactly as it would have in the interrupted
        execution. The optimizer state is applied separately inside the
        training loop, where the optimizer is constructed.

        Args:
            restored: The validated checkpoint.
        """
        meta = restored.meta
        _ = self._prepared.model.load_state_dict(restored.model_state)
        best_val_loss = meta["best_val_loss"]
        self._es_state = EarlyStoppingState(
            best_val_loss=float("inf") if best_val_loss is None else best_val_loss,
            epochs_no_improve=meta["epochs_no_improve"],
        )
        if meta["best_saved"]:
            self._best_checkpoint_path = Path(
                str(_test_hooks.model_dir(self._settings, self._run_id))
            )
        self._total_samples_processed = meta["total_samples_processed"]
        self._total_tokens_processed = meta["total_tokens_processed"]
        self._epoch_summaries = [
            (
                record["epoch"],
                record["train_loss"],
                record["train_ppl"],
                record["val_loss"],
                record["val_ppl"],
            )
            for record in meta["epoch_summaries"]
        ]
        self._elapsed_before = meta["elapsed_seconds"]
        self._training_start_iso = meta["started_at_iso"]
        self._resumed_from_epoch = meta["epochs_completed"]
        restore_rng_states(restored.rng)
        _logger.info(
            "Resuming run from checkpoint",
            extra={
                "category": "training",
                "event": "training_resumed",
                "run_id": self._run_id,
                "resumed_from_epoch": meta["epochs_completed"],
                "global_step": meta["global_step"],
            },
        )

    def _save_epoch_checkpoint(
        self,
        *,
        model: LMModelProto,
        optim: OptimizerProto,
        epochs_completed: int,
        global_step: int,
        last_loss: float,
    ) -> None:
        """Persist the rolling checkpoint at an epoch boundary.

        Args:
            model: The model whose state to persist.
            optim: The optimizer whose state to persist.
            epochs_completed: Number of fully completed epochs.
            global_step: Optimizer steps taken so far.
            last_loss: Training loss at the boundary.
        """
        best = self._es_state["best_val_loss"]
        now = _test_hooks.time_monotonic()
        meta: TrainingCheckpointMeta = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self._run_id,
            "epochs_completed": epochs_completed,
            "global_step": global_step,
            "last_loss": last_loss,
            "best_val_loss": None if best == float("inf") else best,
            "epochs_no_improve": self._es_state["epochs_no_improve"],
            "best_saved": self._best_checkpoint_path is not None,
            "total_samples_processed": self._total_samples_processed,
            "total_tokens_processed": self._total_tokens_processed,
            "elapsed_seconds": self._elapsed_before + (now - self._training_start_time),
            "started_at_iso": self._training_start_iso,
            "epoch_summaries": [
                EpochSummaryRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_ppl=train_ppl,
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                )
                for epoch, train_loss, train_ppl, val_loss, val_ppl in self._epoch_summaries
            ],
            "config": self._cfg,
        }
        _ = save_training_checkpoint(
            self._settings,
            TrainingCheckpoint(
                meta=meta,
                model_state=model.state_dict(),
                optimizer_state=optim.state_dict(),
                rng=capture_rng_states(),
            ),
        )
