"""Test utilities for finetuning strategy tests.

Provides FakeModel implementation for testing strategies without
requiring torch or actual model implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

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

    n_positions = 8


class FakeModel(LMModelProto):
    """Fake model implementing LMModelProto for testing.

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

    @classmethod
    def from_pretrained(cls, path: str) -> LMModelProto:
        """Create a fake model from a path.

        Args:
            path: The path to load from (ignored).

        Returns:
            New FakeModel instance.
        """
        return cls(f"loaded-from-{path}")

    def train(self) -> None:
        """Set model to training mode (no-op)."""
        pass

    def eval(self) -> None:
        """Set model to evaluation mode (no-op)."""
        pass

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
        """Move model to device (no-op).

        Args:
            device: Target device.

        Returns:
            Self.
        """
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

    def state_dict(self: FakeModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: FakeModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self
