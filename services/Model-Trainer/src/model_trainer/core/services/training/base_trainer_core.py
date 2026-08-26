"""BaseTrainer core: construction, device, LR cap, loaders.

The trainer is assembled as a linear chain: core -> observability ->
checkpoints -> loop -> :class:`model_trainer.core.services.training.base_trainer.BaseTrainer`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from platform_core.determinism_record import DeterminismRecord
from platform_core.logging import get_logger
from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import DatasetConfig
from model_trainer.core.contracts.model import (
    EarlyStoppingState,
    ModelTrainConfig,
    PreparedLMModel,
)
from model_trainer.core.services.training.dataloader import DataLoader
from model_trainer.core.services.training.dataset_builder import CausalLMDataset
from model_trainer.core.types import (
    OptimizerCtorProto,
)
from model_trainer.infra.persistence.models import (
    TrainingManifestVersions,
)

_logger = get_logger(__name__)


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

    The build-stamped GIT_COMMIT variable wins: a deployed image carries no
    .git directory, so the subprocess path below can never answer inside a
    container, and every manifest the 2026-08-18 provenance audit archived
    had git_commit null for exactly that reason. The subprocess path remains
    for development checkouts run outside the image.

    Args:
        settings: Application settings containing artifacts_root.
        service_name: Name of the service for logging.

    Returns:
        Git commit hash or None if detection fails.
    """
    import subprocess as _sp

    stamped = _test_hooks.env_git_commit()
    if stamped is not None:
        return stamped

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


class _TrainerCore:
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
    _resume: bool
    # New instance state for enhanced training
    _device: torch.device
    _es_state: EarlyStoppingState
    _best_checkpoint_path: Path | None
    _val_loader: DataLoader | None
    _test_loader: DataLoader | None
    _epoch_summaries: list[tuple[int, float, float, float, float]]
    # Training metrics tracking
    _training_start_time: float
    _training_start_iso: str
    _total_samples_processed: int
    _total_tokens_processed: int
    # Resume state: epoch the current execution continued from (None when
    # training from scratch) and wall-clock seconds consumed by prior
    # executions of this run.
    _resumed_from_epoch: int | None
    _elapsed_before: float

    def __init__(
        self,
        prepared: PreparedLMModel,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        redis_hb: Callable[[float], None],
        cancelled: Callable[[], bool],
        resume: bool,
        progress: (
            Callable[[int, int, float, float, float, float, float | None, float | None], None]
            | None
        ) = None,
        service_name: str = "base-trainer",
        wandb_publisher: WandbPublisher | None = None,
        determinism: DeterminismRecord | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            prepared: Prepared model with tokenizer and config.
            cfg: Training configuration with model_family, model_size, etc.
            settings: Application settings.
            run_id: Unique identifier for this training run.
            redis_hb: Heartbeat callback (called with timestamp every 10 steps).
            cancelled: Callback to check if training was cancelled.
            resume: When True, continue the run from its persisted
                checkpoint; training refuses to start when no valid
                checkpoint exists or the config differs from the one the
                checkpoint records.
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
        self._resume = resume
        self._progress = progress
        self._service_name = service_name
        self._wandb = wandb_publisher
        # What the worker pinned before any CUDA work, carried here so the
        # manifest can record it. None when the caller did not pin -- which
        # the manifest records as "not recorded" rather than as pinned.
        self._determinism = determinism
        self._epoch_summaries: list[tuple[int, float, float, float, float]] = []
        # Initialize training metrics tracking (may be overwritten in train())
        self._training_start_time = 0.0
        self._training_start_iso = ""
        self._total_samples_processed = 0
        self._total_tokens_processed = 0
        self._resumed_from_epoch = None
        self._elapsed_before = 0.0
        # Early-stopping state lives here rather than only in train() so the
        # epoch-boundary checkpoint writer never sees a partial object.
        self._es_state = EarlyStoppingState(
            best_val_loss=float("inf"),
            epochs_no_improve=0,
        )
        self._best_checkpoint_path = None

    def _setup_device(self) -> torch.device:
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
            return _test_hooks.torch_device("cuda")
        return _test_hooks.torch_device("cpu")

    def _apply_lr_cap(self) -> float:
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
        self,
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
        split = _test_hooks.split_corpus(ds_cfg)

        def make_loader(lines: tuple[str, ...], shuffle: bool) -> DataLoader | None:
            if not lines:
                return None
            dataset = CausalLMDataset(
                lines=lines,
                tokenizer=self._prepared.tok_for_dataset,
                max_len=self._prepared.max_seq_len,
                eos_id=self._prepared.eos_id,
                pad_id=self._prepared.pad_id,
                loss_mask_prefix_separator=self._cfg["loss_mask_prefix_separator"],
            )
            return DataLoader(
                dataset,
                batch_size=self._cfg["batch_size"],
                shuffle=shuffle,
                num_workers=self._cfg["data_num_workers"],
                pin_memory=self._cfg["data_pin_memory"],
            )

        train_loader = make_loader(split["train"], shuffle=True)
        val_loader = make_loader(split["validation"], shuffle=False)
        test_loader = make_loader(split["test"], shuffle=False)

        if train_loader is None:
            raise RuntimeError("No training data available")

        # Calculate total batches for progress tracking
        train_batches = len(train_loader)
        val_batches = len(val_loader) if val_loader is not None else 0
        test_batches = len(test_loader) if test_loader is not None else 0
        total_train_steps = train_batches * self._cfg["num_epochs"]

        _logger.info(
            "Data loaders built",
            extra={
                "category": "training",
                "event": "loaders_built",
                "train_lines": len(split["train"]),
                "val_lines": len(split["validation"]),
                "test_lines": len(split["test"]),
                "train_batches": train_batches,
                "val_batches": val_batches,
                "test_batches": test_batches,
                "total_train_steps": total_train_steps,
            },
        )

        return train_loader, val_loader, test_loader
