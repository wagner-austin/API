"""HF LM test doubles: scoring, generation, eval models and loaders."""

from __future__ import annotations

from collections.abc import Generator, Sequence

import torch
from tests.core.services.model.backends.hf_lm.testing import _FakeConfig, _FakeFwd

from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    CausalLMDatasetProto,
    DataLoaderProto,
)
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    LoadStateDictResultProto,
    NamedParameter,
    ParameterLike,
)


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

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing (no-op for fake)."""
        return

    @property
    def config(self) -> ConfigLike:
        """Return config."""
        return _FakeConfig()

    def state_dict(self: FakeScoreModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: FakeScoreModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self


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

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing (no-op for fake)."""
        return

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

    def state_dict(self: FakeGenerateModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: FakeGenerateModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self


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

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing (no-op for fake)."""
        return

    @property
    def config(self) -> ConfigLike:
        """Return config."""
        return _FakeConfig()

    def state_dict(self: FakeEvalModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: FakeEvalModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self


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

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing (no-op for fake)."""
        return

    @property
    def config(self) -> ConfigLike:
        """Return config."""
        return _FakeConfig()

    def state_dict(self: FakeSinglePositionScoreModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: FakeSinglePositionScoreModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self


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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample.

        Args:
            idx: Index.

        Returns:
            Input ids and labels. The real dataset returns labels equal to the
            inputs unless a prefix is masked, so this fake does the same.
        """
        ids = torch.randint(0, 100, (128,))
        return (ids, ids)


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

    def __iter__(self) -> Generator[Sequence[torch.Tensor], None, None]:
        """Iterate over batches of (input_ids, labels).

        Collates the same way torch's default_collate does for a dataset of
        2-tuples: each position is stacked independently, so consumers index
        the result rather than unpacking a single tensor.
        """
        num_samples = len(self._dataset)
        for i in range(0, num_samples, self._batch_size):
            items = [self._dataset[j] for j in range(i, min(i + self._batch_size, num_samples))]
            yield (
                torch.stack([item[0] for item in items]),
                torch.stack([item[1] for item in items]),
            )
