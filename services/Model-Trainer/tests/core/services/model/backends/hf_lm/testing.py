"""Test utilities for HuggingFace LM backend tests.

Provides fake implementations for testing without requiring transformers
or actual model implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch

from model_trainer.core.contracts.finetuning import StrategyName
from model_trainer.core.contracts.model import GenerateConfig, ModelTrainConfig, ScoreConfig
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    LoadStateDictResultProto,
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

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing (no-op for fake)."""
        return

    @property
    def config(self) -> ConfigLike:
        """Return fake config.

        Returns:
            Fake config object.
        """
        return _FakeConfig()

    def state_dict(self: FakeHFModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: FakeHFModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self


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
        "corpus_format": "lines",
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
        "loss_mask_prefix_separator": None,
        "finetuning_strategy": finetuning_strategy,
        "hub_model_id": hub_model_id,
        "lora": None,
        "quantization": None,
        "gguf_export": None,
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


def fake_load_tokenizer(path: str) -> FakeTokenizerHandle:
    """Fake tokenizer loader for testing.

    Args:
        path: Path to tokenizer (ignored in fake).

    Returns:
        FakeTokenizerHandle instance.
    """
    return FakeTokenizerHandle()
