"""Hook protocols for the HF LM backend."""

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
        resume: bool,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the sample at index as (input_ids, labels)."""
        ...


class CreateCausalDatasetFn(Protocol):
    """Protocol for creating CausalLMDataset instances."""

    def __call__(
        self,
        *,
        lines: Sequence[str],
        tokenizer: Encoder,
        max_len: int,
        eos_id: int,
        pad_id: int,
    ) -> CausalLMDatasetProto:
        """Create a causal LM dataset.

        Args:
            lines: Corpus lines to tokenize, already partitioned.
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

    def __iter__(self) -> Generator[Sequence[torch.Tensor], None, None]:
        """Return iterator over batches of (input_ids, labels)."""
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
