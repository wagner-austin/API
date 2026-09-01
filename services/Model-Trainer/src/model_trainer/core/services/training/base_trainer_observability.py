"""BaseTrainer observability: the training manifest and W&B logging."""

from __future__ import annotations

import os
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str

from model_trainer.core.run_fingerprint import capture_run_fingerprint
from model_trainer.core.services.training.base_trainer_core import (
    _gather_lib_versions,
    _maybe_git_commit,
    _TrainerCore,
)
from model_trainer.infra.persistence.models import (
    TrainingManifest,
    TrainingManifestConfig,
    TrainingManifestFull,
    TrainingManifestModelInfo,
    TrainingManifestPerformance,
    TrainingManifestTiming,
    TrainingManifestVersions,
)
from model_trainer.worker.manifest_encoding import encode_training_manifest_full


class _TrainerObservability(_TrainerCore):
    """Manifest writing and W&B logging over the core trainer state."""

    def _write_manifest(
        self,
        *,
        out_dir: str,
        steps: int,
        last_loss: float,
        test_loss: float | None,
        test_ppl: float | None,
        best_val_loss: float | None,
        early_stopped: bool,
        training_duration_sec: float,
        started_at: str,
        completed_at: str,
        peak_gpu_memory_mb: float | None,
        avg_samples_per_sec: float,
        total_tokens_processed: int,
        param_count: int,
        model_size_mb: float,
        vocab_size: int,
    ) -> None:
        """Write training manifest to disk.

        Args:
            out_dir: Output directory for manifest.json.
            steps: Total training steps completed.
            last_loss: Final loss value.
            test_loss: Test set loss (None if no test evaluation).
            test_ppl: Test set perplexity (None if no test evaluation).
            best_val_loss: Best validation loss achieved (None if no validation).
            early_stopped: Whether training stopped early due to no improvement.
            training_duration_sec: Total training time in seconds.
            started_at: ISO 8601 timestamp when training began.
            completed_at: ISO 8601 timestamp when training finished.
            peak_gpu_memory_mb: Maximum GPU memory used (None if CPU).
            avg_samples_per_sec: Average throughput during training.
            total_tokens_processed: Total tokens seen during training.
            param_count: Total trainable parameters in model.
            model_size_mb: Size of saved model on disk in megabytes.
            vocab_size: Tokenizer vocabulary size.
        """
        import platform as _platform

        vers: TrainingManifestVersions = _gather_lib_versions(self._service_name)

        timing: TrainingManifestTiming = {
            "training_duration_sec": training_duration_sec,
            "started_at": started_at,
            "completed_at": completed_at,
        }

        performance: TrainingManifestPerformance = {
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "avg_samples_per_sec": avg_samples_per_sec,
            "total_tokens_processed": total_tokens_processed,
        }

        model_info: TrainingManifestModelInfo = {
            "param_count": param_count,
            "model_size_mb": model_size_mb,
            "vocab_size": vocab_size,
        }

        manifest: TrainingManifest = {
            "run_id": self._run_id,
            "model_family": self._cfg["model_family"],
            "model_size": self._cfg["model_size"],
            "epochs": self._cfg["num_epochs"],
            "batch_size": self._cfg["batch_size"],
            "max_seq_len": self._cfg["max_seq_len"],
            "steps": steps,
            "loss": last_loss,
            "learning_rate": self._cfg["learning_rate"],
            "tokenizer_id": self._cfg["tokenizer_id"],
            "corpus_path": self._cfg["corpus_path"],
            "corpus_format": self._cfg["corpus_format"],
            "holdout_fraction": self._cfg["holdout_fraction"],
            "optimizer": self._cfg["optimizer"],
            "freeze_embed": self._cfg["freeze_embed"],
            "gradient_clipping": self._cfg["gradient_clipping"],
            "seed": self._cfg["seed"],
            "pretrained_run_id": self._cfg["pretrained_run_id"],
            "versions": vers,
            "system": {
                "cpu_count": int(os.cpu_count() or 1),
                "platform": _platform.system(),
                "platform_release": _platform.release(),
                "machine": _platform.machine(),
            },
            # The whole configuration, from the SAME function the scoring path
            # calls, so a training run and a scoring run are comparable by one
            # rule rather than two. It is keyed on the run's DEVICE, not on
            # hardware presence: a cpu run records no card even on a CUDA box,
            # and querying one would needlessly initialise a CUDA context in
            # the writing process. _setup_device already guarantees that
            # device "cuda" implies CUDA is available.
            "fingerprint": capture_run_fingerprint(self._cfg["device"], self._determinism),
            "git_commit": _maybe_git_commit(self._settings, self._service_name),
            "device": self._cfg["device"],
            "precision": self._cfg["precision"],
            "early_stopping_patience": self._cfg["early_stopping_patience"],
            "test_split_ratio": self._cfg["test_split_ratio"],
            "finetune_lr_cap": self._cfg["finetune_lr_cap"],
            "loss_mask_prefix_separator": self._cfg["loss_mask_prefix_separator"],
            "test_loss": test_loss,
            "test_perplexity": test_ppl,
            "best_val_loss": best_val_loss,
            "early_stopped": early_stopped,
            "resumed_from_epoch": self._resumed_from_epoch,
            "timing": timing,
            "performance": performance,
            "model_info": model_info,
            "gguf_export": None,
        }

        cfg_block: TrainingManifestConfig = {
            "model_family": self._cfg["model_family"],
            "model_size": self._cfg["model_size"],
            "max_seq_len": self._cfg["max_seq_len"],
            "num_epochs": self._cfg["num_epochs"],
            "batch_size": self._cfg["batch_size"],
            "learning_rate": self._cfg["learning_rate"],
            "tokenizer_id": self._cfg["tokenizer_id"],
            "corpus_path": self._cfg["corpus_path"],
            "corpus_format": self._cfg["corpus_format"],
            "holdout_fraction": self._cfg["holdout_fraction"],
            "seed": self._cfg["seed"],
            "pretrained_run_id": self._cfg["pretrained_run_id"],
            "freeze_embed": self._cfg["freeze_embed"],
            "gradient_clipping": self._cfg["gradient_clipping"],
            "optimizer": self._cfg["optimizer"],
            "device": self._cfg["device"],
            "precision": self._cfg["precision"],
            "early_stopping_patience": self._cfg["early_stopping_patience"],
            "test_split_ratio": self._cfg["test_split_ratio"],
            "finetune_lr_cap": self._cfg["finetune_lr_cap"],
            "loss_mask_prefix_separator": self._cfg["loss_mask_prefix_separator"],
        }

        full: TrainingManifestFull = {
            "run_id": manifest["run_id"],
            "model_family": manifest["model_family"],
            "model_size": manifest["model_size"],
            "epochs": manifest["epochs"],
            "batch_size": manifest["batch_size"],
            "max_seq_len": manifest["max_seq_len"],
            "steps": manifest["steps"],
            "loss": manifest["loss"],
            "learning_rate": manifest["learning_rate"],
            "tokenizer_id": manifest["tokenizer_id"],
            "corpus_path": manifest["corpus_path"],
            "corpus_format": manifest["corpus_format"],
            "holdout_fraction": manifest["holdout_fraction"],
            "optimizer": manifest["optimizer"],
            "freeze_embed": manifest["freeze_embed"],
            "gradient_clipping": manifest["gradient_clipping"],
            "seed": manifest["seed"],
            "pretrained_run_id": manifest["pretrained_run_id"],
            "versions": manifest["versions"],
            "system": manifest["system"],
            "fingerprint": manifest["fingerprint"],
            "git_commit": manifest["git_commit"],
            "config": cfg_block,
            "device": manifest["device"],
            "precision": manifest["precision"],
            "early_stopping_patience": manifest["early_stopping_patience"],
            "test_split_ratio": manifest["test_split_ratio"],
            "finetune_lr_cap": manifest["finetune_lr_cap"],
            "loss_mask_prefix_separator": manifest["loss_mask_prefix_separator"],
            "test_loss": manifest["test_loss"],
            "test_perplexity": manifest["test_perplexity"],
            "best_val_loss": manifest["best_val_loss"],
            "early_stopped": manifest["early_stopped"],
            "resumed_from_epoch": manifest["resumed_from_epoch"],
            "timing": manifest["timing"],
            "performance": manifest["performance"],
            "model_info": manifest["model_info"],
            "gguf_export": manifest["gguf_export"],
        }

        # Encoded rather than dumped. The fingerprint is not JSON-native -- a
        # DeterminismRecord holds its settings as sorted (name, value) PAIRS
        # -- so dumping the manifest raw writes a list where the decoder
        # requires an object, producing a file the code that wrote it cannot
        # read back.
        payload: JSONObject = encode_training_manifest_full(full)
        Path(out_dir).joinpath("manifest.json").write_text(dump_json_str(payload), encoding="utf-8")

    def _log_wandb_config(self) -> None:
        """Log training configuration to wandb at start of training."""
        if self._wandb is None:
            return
        self._wandb.log_config(
            {
                "run_id": self._run_id,
                "model_family": self._cfg["model_family"],
                "model_size": self._cfg["model_size"],
                "num_epochs": self._cfg["num_epochs"],
                "batch_size": self._cfg["batch_size"],
                "learning_rate": self._cfg["learning_rate"],
                "device": self._cfg["device"],
                "precision": self._cfg["precision"],
                "optimizer": self._cfg["optimizer"],
                "gradient_clipping": self._cfg["gradient_clipping"],
                "freeze_embed": self._cfg["freeze_embed"],
                "early_stopping_patience": self._cfg["early_stopping_patience"],
                "seed": self._cfg["seed"],
                "max_seq_len": self._cfg["max_seq_len"],
                "tokenizer_id": self._cfg["tokenizer_id"],
                "corpus_path": self._cfg["corpus_path"],
                "corpus_format": self._cfg["corpus_format"],
                "holdout_fraction": self._cfg["holdout_fraction"],
                "test_split_ratio": self._cfg["test_split_ratio"],
                "pretrained_run_id": self._cfg["pretrained_run_id"],
                "finetune_lr_cap": self._cfg["finetune_lr_cap"],
            }
        )

    def _log_wandb_step(
        self,
        *,
        step: int,
        epoch: int,
        train_loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
    ) -> None:
        """Log per-step training metrics to wandb."""
        if self._wandb is None:
            return
        self._wandb.log_step(
            {
                "global_step": step,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "grad_norm": grad_norm,
                "samples_per_sec": samples_per_sec,
            }
        )

    def _log_wandb_epoch(
        self,
        *,
        epoch: int,
        train_loss: float,
        train_ppl: float,
        val_loss: float,
        val_ppl: float,
        best_val_loss: float,
        epochs_no_improve: int,
    ) -> None:
        """Log epoch-end metrics with validation results to wandb."""
        if self._wandb is None:
            return
        self._wandb.log_epoch(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
            }
        )

    def _log_wandb_final(
        self,
        *,
        test_loss: float | None,
        test_ppl: float | None,
        early_stopped: bool,
    ) -> None:
        """Log final training metrics and finish wandb run."""
        if self._wandb is None:
            return
        # Build final metrics dict - only include non-None values
        final_metrics: dict[str, float | int | bool] = {"early_stopped": early_stopped}
        if test_loss is not None:
            final_metrics["test_loss"] = test_loss
        if test_ppl is not None:
            final_metrics["test_ppl"] = test_ppl
        self._wandb.log_final(final_metrics)

    def _log_wandb_epoch_table(self) -> None:
        """Log epoch summary table to wandb."""
        if self._wandb is None:
            return
        if not self._epoch_summaries:
            return
        columns = ["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl"]
        data: list[list[float | int]] = [
            [epoch, train_loss, train_ppl, val_loss, val_ppl]
            for epoch, train_loss, train_ppl, val_loss, val_ppl in self._epoch_summaries
        ]
        self._wandb.log_table("epoch_summary", columns, data)

    def _finish_wandb(self) -> None:
        """Finish wandb run after logging all data."""
        if self._wandb is None:
            return
        self._wandb.finish()
