"""Internal hooks for HuggingFace LM backend dependency injection.

Production code sets hooks to real implementations at startup.
Tests set hooks to fakes. No conditionals - call hooks directly.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Literal, Protocol

import torch
from platform_ml.wandb_publisher import WandbPublisher

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import (
    ModelTrainConfig,
    PreparedLMModel,
    TrainOutcome,
)
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoder
from model_trainer.core.types import LMModelProto

ProgressCallback = Callable[
    [int, int, float, float, float, float, float | None, float | None], None
]


class HFTokenizerProto(Protocol):
    """Protocol for HuggingFace tokenizer interface."""

    @property
    def eos_token_id(self) -> int | None:
        """End of sequence token ID."""
        ...

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        ...

    def __len__(self) -> int:
        """Vocabulary size."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode.

        Returns:
            List of token IDs.
        """
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: Token IDs to decode.

        Returns:
            Decoded text string.
        """
        ...

    def convert_tokens_to_ids(self, token: str) -> int:
        """Convert a single token to its ID.

        Args:
            token: Token string to convert.

        Returns:
            Token ID.
        """
        ...


class _HFTokenizerClassProto(Protocol):
    """Protocol for HuggingFace tokenizer class with from_pretrained."""

    def from_pretrained(self, model_id_or_path: str) -> HFTokenizerProto:
        """Load tokenizer from pretrained.

        Args:
            model_id_or_path: HuggingFace model ID or local path.

        Returns:
            Loaded tokenizer instance.
        """
        ...


class HFModelLoader(Protocol):
    """Protocol for loading HuggingFace models from hub or path."""

    def __call__(self, model_id_or_path: str) -> LMModelProto:
        """Load a model from HuggingFace Hub or local path.

        Args:
            model_id_or_path: HuggingFace model ID or local path.

        Returns:
            Loaded language model.
        """
        ...


class HFTokenizerLoader(Protocol):
    """Protocol for loading HuggingFace tokenizers from hub or path."""

    def __call__(self, model_id_or_path: str) -> HFTokenizerProto:
        """Load a tokenizer from HuggingFace Hub or local path.

        Args:
            model_id_or_path: HuggingFace model ID or local path.

        Returns:
            Loaded tokenizer.
        """
        ...


# ============================================================================
# Protocols for train.py
# ============================================================================


class TrainerProto(Protocol):
    """Protocol for trainer instances that can run training."""

    def train(self) -> TrainOutcome:
        """Run training and return outcome."""
        ...


class CreateTrainerFn(Protocol):
    """Protocol for creating trainer instances."""

    def __call__(
        self,
        prepared: PreparedLMModel,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        redis_hb: Callable[[float], None],
        cancelled: Callable[[], bool],
        progress: ProgressCallback | None,
        service_name: str,
        wandb_publisher: WandbPublisher | None,
    ) -> TrainerProto:
        """Create a trainer instance.

        Args:
            prepared: Prepared model to train.
            cfg: Training configuration.
            settings: Application settings.
            run_id: Unique identifier for this training run.
            redis_hb: Heartbeat callback.
            cancelled: Callback to check if training was cancelled.
            progress: Optional progress callback.
            service_name: Name of the service.
            wandb_publisher: Optional W&B publisher.

        Returns:
            Trainer instance.
        """
        ...


# ============================================================================
# Protocols for evaluate.py
# ============================================================================


class TokenizerLoader(Protocol):
    """Protocol for loading tokenizers with automatic backend detection."""

    def __call__(self, path: str) -> TokenizerHandle:
        """Load a tokenizer from path with automatic backend detection.

        Args:
            path: Path to tokenizer artifact directory.

        Returns:
            Loaded tokenizer handle.
        """
        ...


class PreparedModelLoader(Protocol):
    """Protocol for loading prepared models from saved artifacts."""

    def __call__(
        self, model_path: str, tokenizer_handle: TokenizerHandle | None
    ) -> PreparedLMModel:
        """Load a prepared model from path with optional tokenizer handle.

        For HF LM models, the tokenizer_handle is optional (can be None) because
        the HF tokenizer is loaded from hub_model_id stored in metadata.

        Args:
            model_path: Path to model artifacts.
            tokenizer_handle: Optional tokenizer handle (unused for HF LM).

        Returns:
            Prepared model instance.
        """
        ...


class CausalLMDatasetProto(Protocol):
    """Protocol for causal LM dataset."""

    def __len__(self) -> int:
        """Return number of samples."""
        ...

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get sample at index."""
        ...


class CreateCausalDatasetFn(Protocol):
    """Protocol for creating CausalLMDataset instances."""

    def __call__(
        self,
        *,
        files: Sequence[str],
        tokenizer: Encoder,
        max_len: int,
        eos_id: int,
        pad_id: int,
    ) -> CausalLMDatasetProto:
        """Create a causal LM dataset.

        Args:
            files: List of file paths containing training data.
            tokenizer: Encoder for tokenization.
            max_len: Maximum sequence length.
            eos_id: End of sequence token ID.
            pad_id: Padding token ID.

        Returns:
            Dataset instance.
        """
        ...


class DataLoaderProto(Protocol):
    """Protocol for DataLoader instance (iterable over batches)."""

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        """Return iterator over batches."""
        ...


class CreateDataLoaderFn(Protocol):
    """Protocol for creating DataLoader instances."""

    def __call__(
        self,
        dataset: CausalLMDatasetProto,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
    ) -> DataLoaderProto:
        """Create a DataLoader instance.

        Args:
            dataset: Dataset to load from.
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.

        Returns:
            DataLoader instance.
        """
        ...


class ModelDirFn(Protocol):
    """Protocol for getting model directory path."""

    def __call__(self, settings: Settings, run_id: str) -> Path:
        """Get model directory path.

        Args:
            settings: Application settings.
            run_id: Run identifier.

        Returns:
            Path to model directory.
        """
        ...


class EvalDirFn(Protocol):
    """Protocol for getting evaluation directory path."""

    def __call__(self, settings: Settings, run_id: str) -> Path:
        """Get evaluation directory path.

        Args:
            settings: Application settings.
            run_id: Run identifier.

        Returns:
            Path to evaluation directory.
        """
        ...


class AutocastContextFn(Protocol):
    """Protocol for getting autocast context manager."""

    def __call__(
        self, precision: Literal["fp32", "fp16", "bf16"], device_type: str
    ) -> torch.autocast:
        """Get autocast context for precision and device.

        Args:
            precision: The precision to use.
            device_type: The device type.

        Returns:
            Autocast context manager.
        """
        ...


# ============================================================================
# Protocols for generate.py and score.py
# ============================================================================


class ReadTextFileFn(Protocol):
    """Protocol for reading text files."""

    def __call__(self, path: Path) -> str:
        """Read text from a file.

        Args:
            path: Path to file.

        Returns:
            File contents as string.
        """
        ...


class Hooks:
    """Container for HF LM backend test hooks.

    Production code sets these to real implementations.
    Tests set these to fakes for isolation.
    """

    # io.py / prepare.py hooks
    load_hf_model: HFModelLoader | None = None
    load_hf_tokenizer: HFTokenizerLoader | None = None

    # train.py hooks
    create_trainer: CreateTrainerFn | None = None

    # evaluate.py hooks
    load_tokenizer: TokenizerLoader | None = None
    load_prepared_model: PreparedModelLoader | None = None
    create_causal_dataset: CreateCausalDatasetFn | None = None
    create_dataloader: CreateDataLoaderFn | None = None
    get_model_dir: ModelDirFn | None = None
    get_eval_dir: EvalDirFn | None = None

    # generate.py / score.py hooks
    read_text_file: ReadTextFileFn | None = None

    @classmethod
    def reset(cls) -> None:
        """Restore every hook to its default.

        The restoration `reset_hooks()` performs, exposed as a classmethod so
        an autouse fixture can name the container it protects.
        """
        reset_hooks()


def reset_hooks() -> None:
    """Reset all hooks to None for test cleanup."""
    Hooks.load_hf_model = None
    Hooks.load_hf_tokenizer = None
    Hooks.create_trainer = None
    Hooks.load_tokenizer = None
    Hooks.load_prepared_model = None
    Hooks.create_causal_dataset = None
    Hooks.create_dataloader = None
    Hooks.get_model_dir = None
    Hooks.get_eval_dir = None
    Hooks.read_text_file = None


def _default_load_hf_model(model_id_or_path: str) -> LMModelProto:
    """Production implementation for loading HuggingFace models.

    Args:
        model_id_or_path: HuggingFace model ID or local path.

    Returns:
        Loaded language model instance.
    """
    transformers = __import__("transformers", fromlist=["AutoModelForCausalLM"])
    model_cls: type[LMModelProto] = transformers.AutoModelForCausalLM
    model: LMModelProto = model_cls.from_pretrained(model_id_or_path)
    return model


def _default_load_hf_tokenizer(model_id_or_path: str) -> HFTokenizerProto:
    """Production implementation for loading HuggingFace tokenizers.

    Args:
        model_id_or_path: HuggingFace model ID or local path.

    Returns:
        Loaded tokenizer instance.
    """
    transformers = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer_cls: _HFTokenizerClassProto = transformers.AutoTokenizer
    return tokenizer_cls.from_pretrained(model_id_or_path)


def _default_create_trainer(
    prepared: PreparedLMModel,
    cfg: ModelTrainConfig,
    settings: Settings,
    *,
    run_id: str,
    redis_hb: Callable[[float], None],
    cancelled: Callable[[], bool],
    progress: ProgressCallback | None,
    service_name: str,
    wandb_publisher: WandbPublisher | None,
) -> TrainerProto:
    """Production implementation for creating BaseTrainer.

    Args:
        prepared: Prepared model to train.
        cfg: Training configuration.
        settings: Application settings.
        run_id: Unique identifier for this training run.
        redis_hb: Heartbeat callback.
        cancelled: Callback to check if training was cancelled.
        progress: Optional progress callback.
        service_name: Name of the service.
        wandb_publisher: Optional W&B publisher.

    Returns:
        Trainer instance.
    """
    from model_trainer.core.services.training.base_trainer import BaseTrainer

    trainer: TrainerProto = BaseTrainer(
        prepared,
        cfg,
        settings,
        run_id=run_id,
        redis_hb=redis_hb,
        cancelled=cancelled,
        progress=progress,
        service_name=service_name,
        wandb_publisher=wandb_publisher,
    )
    return trainer


def _default_load_tokenizer(path: str) -> TokenizerHandle:
    """Production implementation for loading tokenizers with automatic detection.

    Args:
        path: Path to tokenizer artifact directory.

    Returns:
        Loaded tokenizer handle.
    """
    from model_trainer.core.services.tokenizer.loader import load_tokenizer_from_dir

    return load_tokenizer_from_dir(path)


def _default_load_prepared_model(
    model_path: str, tokenizer_handle: TokenizerHandle | None
) -> PreparedLMModel:
    """Production implementation for loading prepared models.

    For HF LM models, the tokenizer_handle is optional (can be None) because
    the HF tokenizer is loaded from hub_model_id stored in metadata.

    Args:
        model_path: Path to model artifacts.
        tokenizer_handle: Optional tokenizer handle (unused for HF LM).

    Returns:
        Prepared model instance.
    """
    from model_trainer.core.services.model.backends.hf_lm.io import (
        load_prepared_hf_lm_from_handle,
    )

    return load_prepared_hf_lm_from_handle(model_path, tokenizer_handle)


def _default_create_causal_dataset(
    *,
    files: Sequence[str],
    tokenizer: Encoder,
    max_len: int,
    eos_id: int,
    pad_id: int,
) -> CausalLMDatasetProto:
    """Production implementation for creating CausalLMDataset.

    Args:
        files: List of file paths containing training data.
        tokenizer: Encoder for tokenization.
        max_len: Maximum sequence length.
        eos_id: End of sequence token ID.
        pad_id: Padding token ID.

    Returns:
        Dataset instance.
    """
    from model_trainer.core.services.training.dataset_builder import CausalLMDataset

    dataset: CausalLMDatasetProto = CausalLMDataset(
        files=list(files),
        tokenizer=tokenizer,
        max_len=max_len,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    return dataset


class _DataLoaderClassProto(Protocol):
    """Protocol for DataLoader constructor."""

    def __call__(
        self,
        dataset: CausalLMDatasetProto,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
    ) -> DataLoaderProto:
        """Create DataLoader instance.

        Args:
            dataset: Dataset to load from.
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.

        Returns:
            DataLoader instance.
        """
        ...


def _default_create_dataloader(
    dataset: CausalLMDatasetProto,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoaderProto:
    """Production implementation for creating DataLoader.

    Args:
        dataset: Dataset to load from.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of workers.
        pin_memory: Whether to pin memory.

    Returns:
        DataLoader instance.
    """
    torch_data = __import__("torch.utils.data", fromlist=["DataLoader"])
    loader_cls: _DataLoaderClassProto = torch_data.DataLoader
    loader: DataLoaderProto = loader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def _default_get_model_dir(settings: Settings, run_id: str) -> Path:
    """Production implementation for getting model directory.

    Args:
        settings: Application settings.
        run_id: Run identifier.

    Returns:
        Path to model directory.
    """
    from model_trainer.core.infra.paths import model_dir

    return model_dir(settings, run_id)


def _default_get_eval_dir(settings: Settings, run_id: str) -> Path:
    """Production implementation for getting evaluation directory.

    Args:
        settings: Application settings.
        run_id: Run identifier.

    Returns:
        Path to evaluation directory.
    """
    from model_trainer.core.infra.paths import model_eval_dir

    return model_eval_dir(settings, run_id)


def _default_read_text_file(path: Path) -> str:
    """Production implementation for reading text files.

    Args:
        path: Path to file.

    Returns:
        File contents as string.
    """
    return path.read_text(encoding="utf-8")


def init_production_hooks() -> None:
    """Initialize hooks with production implementations.

    Call this at application startup to wire real implementations.
    """
    Hooks.load_hf_model = _default_load_hf_model
    Hooks.load_hf_tokenizer = _default_load_hf_tokenizer
    Hooks.create_trainer = _default_create_trainer
    Hooks.load_tokenizer = _default_load_tokenizer
    Hooks.load_prepared_model = _default_load_prepared_model
    Hooks.create_causal_dataset = _default_create_causal_dataset
    Hooks.create_dataloader = _default_create_dataloader
    Hooks.get_model_dir = _default_get_model_dir
    Hooks.get_eval_dir = _default_get_eval_dir
    Hooks.read_text_file = _default_read_text_file


__all__ = [
    "CausalLMDatasetProto",
    "CreateCausalDatasetFn",
    "CreateDataLoaderFn",
    "CreateTrainerFn",
    "DataLoaderProto",
    "EvalDirFn",
    "HFModelLoader",
    "HFTokenizerLoader",
    "HFTokenizerProto",
    "Hooks",
    "ModelDirFn",
    "PreparedModelLoader",
    "ProgressCallback",
    "ReadTextFileFn",
    "TokenizerLoader",
    "TrainerProto",
    "init_production_hooks",
    "reset_hooks",
]
