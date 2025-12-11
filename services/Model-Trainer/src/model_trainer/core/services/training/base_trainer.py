"""Base trainer for language models.

Provides a unified training loop for all LM backends (GPT2, CharLSTM, etc.)
with consistent behavior for:
- Dataset loading and batching
- Training loop with gradient clipping
- Progress reporting and heartbeat
- Cancellation handling
- Model saving and manifest writing
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Final, Literal, Protocol

import torch
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import DatasetConfig
from model_trainer.core.contracts.model import (
    EarlyStoppingState,
    ModelTrainConfig,
    PreparedLMModel,
    TrainOutcome,
    ValidationMetrics,
)
from model_trainer.core.services.model.backends.gpt2._dl import DataLoader
from model_trainer.core.services.training.dataset_builder import CausalLMDataset
from model_trainer.core.types import (
    LMModelProto,
    OptimizerCtorProto,
    OptimizerProto,
    ParameterLike,
)
from model_trainer.infra.persistence.models import (
    TrainingManifest,
    TrainingManifestConfig,
    TrainingManifestFull,
    TrainingManifestVersions,
)

_logger: Final = get_logger(__name__)


def _get_optimizer_class(name: str) -> OptimizerCtorProto:
    """Get optimizer class by name with typed interface via dynamic import."""
    torch_optim = __import__("torch.optim", fromlist=[name])
    cls: OptimizerCtorProto = getattr(torch_optim, name)
    return cls


# Map optimizer names from config to torch class names
_OPTIMIZER_MAP: dict[str, str] = {
    "adamw": "AdamW",
    "adam": "Adam",
    "sgd": "SGD",
}


def _get_optimizer_for_config(optimizer_name: str) -> OptimizerCtorProto:
    """Get optimizer class for the given config name."""
    torch_cls_name = _OPTIMIZER_MAP[optimizer_name]
    return _get_optimizer_class(torch_cls_name)


# Expose AdamW symbol for tests to monkeypatch optimizer behavior
AdamW: OptimizerCtorProto = _get_optimizer_class("AdamW")


def _gather_lib_versions(service_name: str) -> TrainingManifestVersions:
    """Gather library versions for training manifest.

    Args:
        service_name: Name of the service for logging (e.g. "gpt2-train").

    Returns:
        Dictionary with version strings for torch, transformers, tokenizers, datasets.
    """

    def _v(name: str) -> str:
        version = _test_hooks.pkg_version(name)
        if version == "unknown":
            _logger.warning(
                "%s not available for version detection",
                name,
                extra={
                    "category": "model",
                    "service": service_name,
                    "event": "version_detection_missing",
                    "reason": "package_not_found",
                },
            )
        return version

    return {
        "torch": _v("torch"),
        "transformers": _v("transformers"),
        "tokenizers": _v("tokenizers"),
        "datasets": _v("datasets"),
    }


def _maybe_git_commit(settings: Settings, service_name: str) -> str | None:
    """Attempt to get git commit hash for reproducibility.

    Args:
        settings: Application settings containing artifacts_root.
        service_name: Name of the service for logging.

    Returns:
        Git commit hash or None if detection fails.
    """
    import subprocess as _sp

    try:
        repo_root = Path(settings["app"]["artifacts_root"]).parents[1]
        return (
            _sp.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=_sp.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except (_sp.CalledProcessError, FileNotFoundError, OSError) as e:
        _logger.warning(
            "Git commit detection failed: %s",
            e,
            extra={
                "category": "model",
                "service": service_name,
                "event": "git_commit_detection_failed",
                "reason": "git_rev_parse_failed",
            },
        )
        return None


class BaseTrainer:
    """Base trainer for language models.

    Provides a unified training loop that works with any LM backend.
    Handles dataset loading, training loop, progress reporting, and model saving.
    Now includes early stopping, validation, test evaluation, and gradient norm logging.
    Optionally integrates with Weights & Biases for experiment tracking.
    """

    _prepared: PreparedLMModel
    _cfg: ModelTrainConfig
    _settings: Settings
    _run_id: str
    _redis_hb: Callable[[float], None]
    _cancelled: Callable[[], bool]
    _progress: (
        Callable[[int, int, float, float, float, float, float | None, float | None], None] | None
    )
    _service_name: str
    _wandb: WandbPublisher | None
    # New instance state for enhanced training
    _device: torch.device
    _es_state: EarlyStoppingState
    _best_checkpoint_path: Path | None
    _val_loader: DataLoader | None
    _test_loader: DataLoader | None
    _epoch_summaries: list[tuple[int, float, float, float, float]]

    def __init__(
        self: BaseTrainer,
        prepared: PreparedLMModel,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        redis_hb: Callable[[float], None],
        cancelled: Callable[[], bool],
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        service_name: str = "base-trainer",
        wandb_publisher: WandbPublisher | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            prepared: Prepared model with tokenizer and config.
            cfg: Training configuration with model_family, model_size, etc.
            settings: Application settings.
            run_id: Unique identifier for this training run.
            redis_hb: Heartbeat callback (called with timestamp every 10 steps).
            cancelled: Callback to check if training was cancelled.
            progress: Optional callback for progress updates
                (step, epoch, loss, ppl, grad_norm, samples_per_sec, val_loss, val_ppl).
            service_name: Service name for logging.
            wandb_publisher: Optional WandbPublisher for experiment tracking.
        """
        self._prepared = prepared
        self._cfg = cfg
        self._settings = settings
        self._run_id = run_id
        self._redis_hb = redis_hb
        self._cancelled = cancelled
        self._progress = progress
        self._service_name = service_name
        self._wandb = wandb_publisher
        self._epoch_summaries: list[tuple[int, float, float, float, float]] = []

    def train(self: BaseTrainer) -> TrainOutcome:
        """Execute the training loop with early stopping and validation.

        Returns:
            TrainOutcome with loss, perplexity, steps, output directory,
            cancellation status, test metrics, and early stopping info.
        """
        torch.manual_seed(self._cfg["seed"])
        random.seed(self._cfg["seed"])

        # 1. Setup device (NEW - GPU support)
        self._device = self._setup_device()

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

        # 5. Run training loop (UPDATED - returns more info)
        last_loss, step, was_cancelled, early_stopped = self._run_training_loop(
            model, train_loader, effective_lr
        )

        out_dir = str(_test_hooks.model_dir(self._settings, self._run_id))
        os.makedirs(out_dir, exist_ok=True)

        # 6. Save checkpoint if not cancelled and no best was saved
        if not was_cancelled and self._best_checkpoint_path is None:
            self._prepared.model.save_pretrained(out_dir)

        # 7. Run test evaluation (NEW)
        test_loss: float | None = None
        test_ppl: float | None = None
        if not was_cancelled and self._test_loader is not None:
            test_metrics = self._run_evaluation(self._test_loader)
            test_loss = test_metrics["val_loss"]
            test_ppl = test_metrics["val_ppl"]
            _logger.info(
                "Test evaluation completed",
                extra={
                    "category": "training",
                    "event": "test_evaluation",
                    "test_loss": test_loss,
                    "test_ppl": test_ppl,
                },
            )

        # Get best val loss (may be inf if no validation was done)
        best_val_loss: float | None = None
        if self._es_state["best_val_loss"] < float("inf"):
            best_val_loss = self._es_state["best_val_loss"]

        self._write_manifest(
            out_dir=out_dir,
            steps=step,
            last_loss=last_loss,
            test_loss=test_loss,
            test_ppl=test_ppl,
            best_val_loss=best_val_loss,
            early_stopped=early_stopped,
        )

        # Log final metrics and epoch table to wandb
        self._log_wandb_final(
            test_loss=test_loss,
            test_ppl=test_ppl,
            early_stopped=early_stopped,
        )
        self._log_wandb_epoch_table()
        self._finish_wandb()

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

    def _setup_device(self: BaseTrainer) -> torch.device:
        """Setup training device based on config.

        Returns:
            torch.device configured for training.

        Raises:
            RuntimeError: If CUDA requested but not available.
        """
        device_str = self._cfg["device"]
        if device_str == "cuda":
            if not _test_hooks.cuda_is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device("cuda")
        return torch.device("cpu")

    def _apply_lr_cap(self: BaseTrainer) -> float:
        """Apply LR cap when fine-tuning from pretrained model.

        Returns:
            Effective learning rate (capped if fine-tuning).
        """
        lr = self._cfg["learning_rate"]
        if self._cfg["pretrained_run_id"] is not None:
            cap = self._cfg["finetune_lr_cap"]
            effective_lr = min(lr, cap)
            if effective_lr < lr:
                _logger.info(
                    "LR capped for fine-tuning",
                    extra={
                        "category": "training",
                        "event": "lr_cap_applied",
                        "original_lr": lr,
                        "capped_lr": effective_lr,
                    },
                )
            return effective_lr
        return lr

    def _build_all_loaders(
        self: BaseTrainer,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """Build train, val, and test data loaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
            Val and test loaders may be None if no data available.

        Raises:
            RuntimeError: If no training data available.
        """
        ds_cfg = DatasetConfig(
            corpus_path=self._cfg["corpus_path"],
            holdout_fraction=self._cfg["holdout_fraction"],
            test_split_ratio=self._cfg["test_split_ratio"],
        )
        train_files, val_files, test_files = _test_hooks.split_corpus_files(ds_cfg)

        def make_loader(files: list[str], shuffle: bool) -> DataLoader | None:
            if not files:
                return None
            dataset = CausalLMDataset(
                files=files,
                tokenizer=self._prepared.tok_for_dataset,
                max_len=self._prepared.max_seq_len,
                eos_id=self._prepared.eos_id,
                pad_id=self._prepared.pad_id,
            )
            return DataLoader(
                dataset,
                batch_size=self._cfg["batch_size"],
                shuffle=shuffle,
                num_workers=self._cfg["data_num_workers"],
                pin_memory=self._cfg["data_pin_memory"],
            )

        train_loader = make_loader(train_files, shuffle=True)
        val_loader = make_loader(val_files, shuffle=False)
        test_loader = make_loader(test_files, shuffle=False)

        if train_loader is None:
            raise RuntimeError("No training data available")

        _logger.info(
            "Data loaders built",
            extra={
                "category": "training",
                "event": "loaders_built",
                "train_files": len(train_files),
                "val_files": len(val_files),
                "test_files": len(test_files),
            },
        )

        return train_loader, val_loader, test_loader

    def _run_evaluation(self: BaseTrainer, loader: DataLoader) -> ValidationMetrics:
        """Run evaluation on given loader.

        Uses the same precision as training for consistent metrics.

        Args:
            loader: DataLoader to evaluate on.

        Returns:
            ValidationMetrics with loss and perplexity.
        """
        model = self._prepared.model
        model.eval()

        total_loss = 0.0
        num_batches = 0
        device_str = str(self._device)

        # Use same precision as training for consistent metrics
        precision = self._cfg["precision"]
        autocast_ctx = _get_autocast_context(precision, self._device)

        with torch.no_grad():
            for batch in loader:
                inputs = batch.to(device_str)
                with autocast_ctx:
                    outputs = model.forward(input_ids=inputs, labels=inputs)
                total_loss += float(outputs.loss.item())
                num_batches += 1

        model.train()
        avg_loss = total_loss / max(1, num_batches)
        avg_ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")

        return ValidationMetrics(val_loss=avg_loss, val_ppl=avg_ppl)

    def _save_best_checkpoint(self: BaseTrainer) -> None:
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

    def _run_training_loop(
        self: BaseTrainer,
        model: LMModelProto,
        dataloader: DataLoader,
        effective_lr: float,
    ) -> tuple[float, int, bool, bool]:
        """Run the main training loop with early stopping.

        Args:
            model: The language model to train.
            dataloader: Training data loader.
            effective_lr: Learning rate (potentially capped for fine-tuning).

        Returns:
            Tuple of (final_loss, total_steps, was_cancelled, early_stopped).
        """
        optimizer_cls = _get_optimizer_for_config(self._cfg["optimizer"])
        optim = optimizer_cls(model.parameters(), lr=effective_lr)
        step = 0
        last_loss = 0.0
        was_cancelled = False
        early_stopped = False
        device_str = str(self._device)

        for epoch in range(self._cfg["num_epochs"]):
            epoch_step_start = step
            last_loss, step, was_cancelled, avg_grad_norm = self._train_one_epoch(
                model=model,
                dataloader=dataloader,
                optim=optim,
                epoch=epoch,
                device=device_str,
                start_step=step,
            )
            if was_cancelled:
                break

            # Report progress for empty epochs
            if self._progress is not None and step == epoch_step_start:
                ppl = float(math.exp(last_loss)) if last_loss < 20 else float("inf")
                self._progress(step, epoch, last_loss, ppl, 0.0, 0.0, None, None)

            # Run validation after each epoch
            if self._val_loader is not None:
                val_metrics = self._run_evaluation(self._val_loader)
                _logger.info(
                    "Validation completed",
                    extra={
                        "category": "training",
                        "event": "validation",
                        "epoch": epoch,
                        "val_loss": val_metrics["val_loss"],
                        "val_ppl": val_metrics["val_ppl"],
                        "avg_grad_norm": avg_grad_norm,
                    },
                )

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
                    break

        return last_loss, step, was_cancelled, early_stopped

    def _train_one_epoch(
        self: BaseTrainer,
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

            inputs = batch.to(device)

            # Forward pass with autocast (no-op for fp32)
            with autocast_ctx:
                outputs = model.forward(input_ids=inputs, labels=inputs)
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

    def _write_manifest(
        self: BaseTrainer,
        *,
        out_dir: str,
        steps: int,
        last_loss: float,
        test_loss: float | None,
        test_ppl: float | None,
        best_val_loss: float | None,
        early_stopped: bool,
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
        """
        import platform as _platform

        vers: TrainingManifestVersions = _gather_lib_versions(self._service_name)

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
            "git_commit": _maybe_git_commit(self._settings, self._service_name),
            "device": self._cfg["device"],
            "precision": self._cfg["precision"],
            "early_stopping_patience": self._cfg["early_stopping_patience"],
            "test_split_ratio": self._cfg["test_split_ratio"],
            "finetune_lr_cap": self._cfg["finetune_lr_cap"],
            "test_loss": test_loss,
            "test_perplexity": test_ppl,
            "best_val_loss": best_val_loss,
            "early_stopped": early_stopped,
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
            "holdout_fraction": manifest["holdout_fraction"],
            "optimizer": manifest["optimizer"],
            "freeze_embed": manifest["freeze_embed"],
            "gradient_clipping": manifest["gradient_clipping"],
            "seed": manifest["seed"],
            "pretrained_run_id": manifest["pretrained_run_id"],
            "versions": manifest["versions"],
            "system": manifest["system"],
            "git_commit": manifest["git_commit"],
            "config": cfg_block,
            "device": manifest["device"],
            "precision": manifest["precision"],
            "early_stopping_patience": manifest["early_stopping_patience"],
            "test_split_ratio": manifest["test_split_ratio"],
            "finetune_lr_cap": manifest["finetune_lr_cap"],
            "test_loss": manifest["test_loss"],
            "test_perplexity": manifest["test_perplexity"],
            "best_val_loss": manifest["best_val_loss"],
            "early_stopped": manifest["early_stopped"],
        }

        Path(out_dir).joinpath("manifest.json").write_text(dump_json_str(full), encoding="utf-8")

    def _log_wandb_config(self: BaseTrainer) -> None:
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
                "holdout_fraction": self._cfg["holdout_fraction"],
                "test_split_ratio": self._cfg["test_split_ratio"],
                "pretrained_run_id": self._cfg["pretrained_run_id"],
                "finetune_lr_cap": self._cfg["finetune_lr_cap"],
            }
        )

    def _log_wandb_step(
        self: BaseTrainer,
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
        self: BaseTrainer,
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
        self: BaseTrainer,
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

    def _log_wandb_epoch_table(self: BaseTrainer) -> None:
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

    def _finish_wandb(self: BaseTrainer) -> None:
        """Finish wandb run after logging all data."""
        if self._wandb is None:
            return
        self._wandb.finish()


class _GradScalerProto(Protocol):
    """Protocol for torch.cuda.amp.GradScaler."""

    def scale(self, loss: torch.Tensor) -> torch.Tensor: ...
    def unscale_(self, optimizer: OptimizerProto) -> None: ...
    def step(self, optimizer: OptimizerProto) -> None: ...
    def update(self) -> None: ...


class _ClipGradNormProto(Protocol):
    """Protocol for torch.nn.utils.clip_grad_norm_ function."""

    def __call__(
        self,
        parameters: Sequence[ParameterLike],
        max_norm: float,
    ) -> torch.Tensor: ...


def _get_clip_grad_norm() -> _ClipGradNormProto:
    """Get torch.nn.utils.clip_grad_norm_ with typed interface."""
    torch_nn_utils = __import__("torch.nn.utils", fromlist=["clip_grad_norm_"])
    fn: _ClipGradNormProto = torch_nn_utils.clip_grad_norm_
    return fn


def _clip_grad_norm(parameters: Sequence[ParameterLike], *, max_norm: float) -> None:
    """Clip gradients of model parameters.

    Args:
        parameters: Model parameters from model.parameters().
        max_norm: Maximum gradient norm.
    """
    clip_fn = _get_clip_grad_norm()
    _ = clip_fn(parameters, max_norm)


def _clip_grad_norm_with_return(parameters: Sequence[ParameterLike], *, max_norm: float) -> float:
    """Clip gradients of model parameters and return the total norm before clipping.

    Args:
        parameters: Model parameters from model.parameters().
        max_norm: Maximum gradient norm.

    Returns:
        Total gradient norm before clipping (as float).
    """
    clip_fn = _get_clip_grad_norm()
    total_norm = clip_fn(parameters, max_norm)
    return float(total_norm.item())


def _freeze_embeddings(model: LMModelProto) -> None:
    """Freeze embedding layer parameters for fine-tuning.

    Attempts to find and freeze embedding layers using common naming conventions.
    Works with transformers models (wte, embed_tokens) and custom models (embedding).

    Args:
        model: The language model with an embedding layer.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        # Match common embedding layer names across different architectures
        if any(
            embed_name in name.lower()
            for embed_name in ("wte", "embed_tokens", "embedding", "word_embedding")
        ):
            param.requires_grad = False
            frozen_count += 1
    _logger.info(
        "Froze %d embedding parameters",
        frozen_count,
        extra={"category": "model", "event": "freeze_embeddings"},
    )


def _get_autocast_context(
    precision: Literal["fp32", "fp16", "bf16"], device: torch.device
) -> AbstractContextManager[None]:
    """Get autocast context manager based on precision and device.

    Args:
        precision: The precision to use.
        device: The device (cpu or cuda).

    Returns:
        A context manager for autocast, or nullcontext for fp32.
    """
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    # Get autocast from torch.amp (PyTorch 2.0+ API)
    torch_amp = __import__("torch.amp", fromlist=["autocast"])
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    ctx: AbstractContextManager[None] = torch_amp.autocast(device_type="cuda", dtype=dtype)
    return ctx


def _create_grad_scaler() -> _GradScalerProto:
    """Create a GradScaler for fp16 mixed precision training.

    Returns:
        A GradScaler instance for scaling gradients.
    """
    torch_amp = __import__("torch.amp", fromlist=["GradScaler"])
    scaler: _GradScalerProto = torch_amp.GradScaler()
    return scaler
