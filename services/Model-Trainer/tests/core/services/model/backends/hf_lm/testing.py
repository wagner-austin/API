"""Test utilities for HuggingFace LM backend tests.

Provides fake implementations for testing without requiring transformers
or actual model implementations.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Literal

import torch

from model_trainer.core.contracts.finetuning import StrategyName
from model_trainer.core.contracts.model import GenerateConfig, ModelTrainConfig, ScoreConfig
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    CausalLMDatasetProto,
    DataLoaderProto,
)
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    NamedParameter,
    ParameterLike,
)


class _FakeFwd(ForwardOutProto):
    """Fake forward output for testing."""

    @property
    def loss(self) -> torch.Tensor:
        """Return a zero loss tensor."""
        return torch.tensor(0.0)


class _FakeConfig(ConfigLike):
    """Fake config for testing."""

    max_position_embeddings = 512


class FakeHFModel(LMModelProto):
    """Fake HuggingFace model for testing.

    Attributes:
        name: Optional name for tracking in tests.
        _save_path: Path where save_pretrained was called.
    """

    def __init__(self, name: str = "test") -> None:
        """Initialize fake model.

        Args:
            name: Optional name for tracking in tests.
        """
        self.name = name
        self._save_path: str | None = None
        self._device: str = "cpu"

    @classmethod
    def from_pretrained(cls, path: str) -> LMModelProto:
        """Create a fake model from a path.

        Args:
            path: The path to load from.

        Returns:
            New FakeHFModel instance.
        """
        return cls(f"loaded-from-{path}")

    def train(self) -> None:
        """Set model to training mode (no-op)."""

    def eval(self) -> None:
        """Set model to evaluation mode (no-op)."""

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Fake forward pass.

        Args:
            input_ids: Input token IDs.
            labels: Target labels.

        Returns:
            Fake forward output with zero loss.
        """
        return _FakeFwd()

    def parameters(self) -> Sequence[ParameterLike]:
        """Return empty parameter sequence.

        Returns:
            Empty list of parameters.
        """
        return []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return empty named parameter sequence.

        Returns:
            Empty list of named parameters.
        """
        return []

    def to(self, device: str) -> LMModelProto:
        """Move model to device.

        Args:
            device: Target device.

        Returns:
            Self.
        """
        self._device = device
        return self

    def save_pretrained(self, out_dir: str) -> None:
        """Fake save method.

        Args:
            out_dir: Output directory.
        """
        self._save_path = out_dir
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> ConfigLike:
        """Return fake config.

        Returns:
            Fake config object.
        """
        return _FakeConfig()


class FakeHFTokenizer:
    """Fake HuggingFace tokenizer for testing."""

    def __init__(self, vocab_size: int = 100) -> None:
        """Initialize fake tokenizer.

        Args:
            vocab_size: Size of vocabulary.
        """
        self._vocab_size = vocab_size

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return 0

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return 1

    def __len__(self) -> int:
        """Vocabulary size."""
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode.

        Returns:
            List of token IDs (fake encoding).
        """
        return [ord(c) % self._vocab_size for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: Token IDs to decode.

        Returns:
            Decoded text string (fake decoding).
        """
        return "".join(chr(i + 32) for i in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        """Convert a single token to its ID.

        Args:
            token: Token string to convert.

        Returns:
            Token ID.
        """
        return ord(token[0]) % self._vocab_size if token else 0


def make_test_config(
    *,
    finetuning_strategy: StrategyName = "full",
    hub_model_id: str | None = "test/model",
    tokenizer_id: str | None = "test-tok",
) -> ModelTrainConfig:
    """Create a minimal ModelTrainConfig for testing.

    Args:
        finetuning_strategy: Strategy to use.
        hub_model_id: HuggingFace model ID.
        tokenizer_id: Tokenizer ID. None for hf_lm (uses HF tokenizer).

    Returns:
        Test configuration.
    """
    return {
        "model_family": "gpt2",
        "model_size": "small",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "tokenizer_id": tokenizer_id,
        "corpus_path": "/tmp/corpus",
        "holdout_fraction": 0.1,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "precision": "fp32",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 3,
        "test_split_ratio": 0.1,
        "finetune_lr_cap": 0.0001,
        "finetuning_strategy": finetuning_strategy,
        "hub_model_id": hub_model_id,
        "lora": None,
        "quantization": None,
        "unsloth": None,
    }


class FakeEncoder:
    """Fake encoder for testing."""

    def __init__(self, *, decode_result: str | None = None) -> None:
        """Initialize.

        Args:
            decode_result: Fixed string to return from decode (for testing stop sequences).
        """
        self._decode_result = decode_result

    def encode(self, text: str) -> _FakeEncoded:
        """Encode text.

        Args:
            text: Text to encode.

        Returns:
            Fake encoded result.
        """
        ids = [ord(c) % 100 for c in text]
        return _FakeEncoded(ids)

    def token_to_id(self, token: str) -> int | None:
        """Convert token to ID.

        Args:
            token: Token string.

        Returns:
            Token ID.
        """
        return ord(token[0]) % 100 if token else None

    def get_vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Vocabulary size.
        """
        return 100

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs.

        Args:
            ids: Token IDs.

        Returns:
            Decoded string.
        """
        if self._decode_result is not None:
            return self._decode_result
        return "".join(chr((i % 26) + 97) for i in ids)


class _FakeEncoded:
    """Fake encoded output."""

    def __init__(self, ids: list[int]) -> None:
        """Initialize.

        Args:
            ids: Token IDs.
        """
        self._ids = ids

    @property
    def ids(self) -> list[int]:
        """Token IDs."""
        return self._ids


class FakeTokenizerHandle:
    """Fake tokenizer handle for testing."""

    def encode(self, text: str) -> list[int]:
        """Encode text.

        Args:
            text: Text to encode.

        Returns:
            Token IDs.
        """
        return [ord(c) % 100 for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs.

        Args:
            ids: Token IDs.

        Returns:
            Decoded string.
        """
        return "".join(chr((i % 26) + 97) for i in ids)

    def token_to_id(self, token: str) -> int | None:
        """Convert token to ID.

        Args:
            token: Token string.

        Returns:
            Token ID.
        """
        return ord(token[0]) % 100 if token else None

    def get_vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Vocabulary size.
        """
        return 100


class FakeScoreModel(LMModelProto):
    """Fake model for scoring tests."""

    def __init__(self, vocab_size: int = 100, seq_len: int = 10) -> None:
        """Initialize.

        Args:
            vocab_size: Vocabulary size.
            seq_len: Output sequence length.
        """
        self._vocab_size = vocab_size
        self._seq_len = seq_len
        self._device = "cpu"
        self._save_path: str | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> LMModelProto:
        """Load from path.

        Args:
            path: Path to load from.

        Returns:
            New instance.
        """
        return cls()

    def train(self) -> None:
        """Set to training mode."""

    def eval(self) -> None:
        """Set to eval mode."""

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Fake forward pass returning logits.

        Args:
            input_ids: Input token IDs.
            labels: Labels.

        Returns:
            Forward output with logits.
        """
        batch_size = int(input_ids.size(0))
        seq_len = int(input_ids.size(1))
        return _FakeLogitsOut(batch_size, seq_len, self._vocab_size)

    def parameters(self) -> Sequence[ParameterLike]:
        """Return empty parameters."""
        return []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return empty named parameters."""
        return []

    def to(self, device: str) -> LMModelProto:
        """Move to device.

        Args:
            device: Device name.

        Returns:
            Self.
        """
        self._device = device
        return self

    def save_pretrained(self, out_dir: str) -> None:
        """Save to directory.

        Args:
            out_dir: Output directory.
        """
        self._save_path = out_dir

    @property
    def config(self) -> ConfigLike:
        """Return config."""
        return _FakeConfig()


class _FakeLogitsOut(ForwardOutProto):
    """Fake forward output with logits."""

    def __init__(self, batch_size: int = 1, seq_len: int = 10, vocab_size: int = 100) -> None:
        """Initialize.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.
            vocab_size: Vocabulary size.
        """
        self._logits = torch.randn(batch_size, seq_len, vocab_size)

    @property
    def loss(self) -> torch.Tensor:
        """Return loss tensor."""
        return torch.tensor(0.5)

    @property
    def logits(self) -> torch.Tensor:
        """Return logits."""
        return self._logits


class FakeGenerateModel(LMModelProto):
    """Fake model with generate method for testing."""

    def __init__(
        self,
        eos_id: int = 0,
        *,
        include_eos: bool = True,
        output_tokens: list[int] | None = None,
    ) -> None:
        """Initialize.

        Args:
            eos_id: EOS token ID to include in output.
            include_eos: Whether to include EOS in generated tokens.
            output_tokens: Custom output tokens (overrides default generation).
        """
        self._eos_id = eos_id
        self._include_eos = include_eos
        self._output_tokens = output_tokens
        self._device = "cpu"
        self._save_path: str | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> LMModelProto:
        """Load from path.

        Args:
            path: Path to load from.

        Returns:
            New instance.
        """
        return cls()

    def train(self) -> None:
        """Set to training mode."""

    def eval(self) -> None:
        """Set to eval mode."""

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Forward pass.

        Args:
            input_ids: Input token IDs.
            labels: Labels.

        Returns:
            Forward output.
        """
        return _FakeFwd()

    def parameters(self) -> Sequence[ParameterLike]:
        """Return empty parameters."""
        return []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return empty named parameters."""
        return []

    def to(self, device: str) -> LMModelProto:
        """Move to device.

        Args:
            device: Device name.

        Returns:
            Self.
        """
        self._device = device
        return self

    def save_pretrained(self, out_dir: str) -> None:
        """Save to directory.

        Args:
            out_dir: Output directory.
        """
        self._save_path = out_dir

    @property
    def config(self) -> ConfigLike:
        """Return config."""
        return _FakeConfig()

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Generate fake output.

        Args:
            input_ids: Input tensor.
            max_new_tokens: Max new tokens.
            do_sample: Whether to sample.
            temperature: Temperature.
            top_k: Top-k.
            top_p: Top-p.
            num_return_sequences: Number of sequences.
            eos_token_id: EOS token ID.
            pad_token_id: Pad token ID.

        Returns:
            Generated token IDs.
        """
        batch_size = num_return_sequences
        if self._output_tokens is not None:
            token_row = self._output_tokens
        elif self._include_eos:
            token_row = [42, 43, self._eos_id, 44]
        else:
            token_row = [42, 43, 45, 44]
        token_rows: list[list[int]] = [token_row for _ in range(batch_size)]
        new_tokens = torch.tensor(token_rows)
        return torch.cat([input_ids.expand(batch_size, -1), new_tokens], dim=1)


class FakeEvalModel(LMModelProto):
    """Fake model for evaluation tests."""

    def __init__(self, loss_value: float = 0.5) -> None:
        """Initialize.

        Args:
            loss_value: Loss value to return on forward.
        """
        self._loss_value = loss_value
        self._device = "cpu"
        self._save_path: str | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> LMModelProto:
        """Load from path.

        Args:
            path: Path to load from.

        Returns:
            New instance.
        """
        return cls()

    def train(self) -> None:
        """Set to training mode."""

    def eval(self) -> None:
        """Set to eval mode."""

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Forward pass returning loss.

        Args:
            input_ids: Input token IDs.
            labels: Labels.

        Returns:
            Forward output with loss.
        """
        return _FakeLossOut(self._loss_value)

    def parameters(self) -> Sequence[ParameterLike]:
        """Return empty parameters."""
        return []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return empty named parameters."""
        return []

    def to(self, device: str) -> LMModelProto:
        """Move to device.

        Args:
            device: Device name.

        Returns:
            Self.
        """
        self._device = device
        return self

    def save_pretrained(self, out_dir: str) -> None:
        """Save to directory.

        Args:
            out_dir: Output directory.
        """
        self._save_path = out_dir

    @property
    def config(self) -> ConfigLike:
        """Return config."""
        return _FakeConfig()


class _FakeLossOut(ForwardOutProto):
    """Fake forward output with loss."""

    def __init__(self, loss_value: float = 0.5) -> None:
        """Initialize.

        Args:
            loss_value: Loss value to return.
        """
        self._loss = torch.tensor(loss_value)

    @property
    def loss(self) -> torch.Tensor:
        """Return loss tensor."""
        return self._loss


class _FakeSinglePositionLogitsOut(ForwardOutProto):
    """Fake forward output with single-position logits (edge case)."""

    def __init__(self, vocab_size: int = 100) -> None:
        """Initialize.

        Args:
            vocab_size: Vocabulary size.
        """
        self._logits = torch.randn(1, 1, vocab_size)

    @property
    def loss(self) -> torch.Tensor:
        """Return loss tensor."""
        return torch.tensor(0.0)

    @property
    def logits(self) -> torch.Tensor:
        """Return logits with single position."""
        return self._logits


class FakeSinglePositionScoreModel(LMModelProto):
    """Fake model that returns logits with only 1 position (edge case)."""

    def __init__(self, vocab_size: int = 100) -> None:
        """Initialize.

        Args:
            vocab_size: Vocabulary size.
        """
        self._vocab_size = vocab_size
        self._device = "cpu"
        self._save_path: str | None = None

    @classmethod
    def from_pretrained(cls, path: str) -> LMModelProto:
        """Load from path.

        Args:
            path: Path to load from.

        Returns:
            New instance.
        """
        return cls()

    def train(self) -> None:
        """Set to training mode."""

    def eval(self) -> None:
        """Set to eval mode."""

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Forward pass returning logits with only 1 position.

        Args:
            input_ids: Input token IDs.
            labels: Labels.

        Returns:
            Forward output with single-position logits.
        """
        return _FakeSinglePositionLogitsOut(self._vocab_size)

    def parameters(self) -> Sequence[ParameterLike]:
        """Return empty parameters."""
        return []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return empty named parameters."""
        return []

    def to(self, device: str) -> LMModelProto:
        """Move to device.

        Args:
            device: Device name.

        Returns:
            Self.
        """
        self._device = device
        return self

    def save_pretrained(self, out_dir: str) -> None:
        """Save to directory.

        Args:
            out_dir: Output directory.
        """
        self._save_path = out_dir

    @property
    def config(self) -> ConfigLike:
        """Return config."""
        return _FakeConfig()


class FakeDataset(CausalLMDatasetProto):
    """Fake dataset for testing."""

    def __init__(self, num_samples: int = 4) -> None:
        """Initialize.

        Args:
            num_samples: Number of samples.
        """
        self._num_samples = num_samples

    def __len__(self) -> int:
        """Return number of samples."""
        return self._num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get sample.

        Args:
            idx: Index.

        Returns:
            Sample tensor.
        """
        return torch.randint(0, 100, (128,))


class FakeDataLoader(DataLoaderProto):
    """Fake data loader for testing."""

    def __init__(self, dataset: CausalLMDatasetProto, batch_size: int = 2) -> None:
        """Initialize.

        Args:
            dataset: Dataset to iterate over.
            batch_size: Batch size.
        """
        self._dataset = dataset
        self._batch_size = batch_size

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        """Iterate over batches."""
        num_samples = len(self._dataset)
        for i in range(0, num_samples, self._batch_size):
            batch = torch.stack(
                [self._dataset[j] for j in range(i, min(i + self._batch_size, num_samples))]
            )
            yield batch


def make_score_config(
    *,
    text: str | None = "Hello world",
    path: str | None = None,
    detail_level: Literal["summary", "per_char"] = "summary",
    top_k: int | None = None,
    seed: int | None = 42,
) -> ScoreConfig:
    """Create a ScoreConfig for testing.

    Args:
        text: Text to score.
        path: Path to text file.
        detail_level: Detail level.
        top_k: Top-k predictions.
        seed: Random seed.

    Returns:
        ScoreConfig for testing.
    """
    return {
        "text": text,
        "path": path,
        "detail_level": detail_level,
        "top_k": top_k,
        "seed": seed,
    }


def make_generate_config(
    *,
    prompt_text: str | None = "Hello",
    prompt_path: str | None = None,
    max_new_tokens: int = 10,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    stop_on_eos: bool = True,
    stop_sequences: Sequence[str] | None = None,
    seed: int | None = 42,
    num_return_sequences: int = 1,
) -> GenerateConfig:
    """Create a GenerateConfig for testing.

    Args:
        prompt_text: Text prompt.
        prompt_path: Path to prompt file.
        max_new_tokens: Max new tokens.
        temperature: Temperature.
        top_k: Top-k.
        top_p: Top-p.
        stop_on_eos: Whether to stop on EOS.
        stop_sequences: Stop sequences.
        seed: Random seed.
        num_return_sequences: Number of sequences.

    Returns:
        GenerateConfig for testing.
    """
    return {
        "prompt_text": prompt_text,
        "prompt_path": prompt_path,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_on_eos": stop_on_eos,
        "stop_sequences": list(stop_sequences) if stop_sequences is not None else [],
        "seed": seed,
        "num_return_sequences": num_return_sequences,
    }
