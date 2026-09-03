"""Test utilities for finetuning strategy tests.

Provides FakeModel implementation for testing strategies without
requiring torch or actual model implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from typing_extensions import TypedDict

from model_trainer.core.types import (
    CacheCarryingOutProto,
    ConfigLike,
    ForwardOutProto,
    KVCacheProto,
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
        # One instance, held. A real model's ``config`` is an attribute, not a
        # freshly built object, and a fake that rebuilt it per access would
        # make "this model's config reached the caller" untestable.
        self._config = _FakeConfig()

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
        return self._config

    def state_dict(self: FakeModel) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(
        self: FakeModel, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        _ = state_dict
        return self


class RecordedCall(TypedDict):
    """One call a fake model received, with every argument typed.

    A mapping of ``object`` would make every assertion on a recorded call a
    narrowing exercise, and the point of recording is to assert on the values.

    Attributes:
        input_ids: Token ids the caller passed.
        labels: Targets, or None when the call computed no loss.
        past_key_values: The prefix cache, or None.
        attention_mask: The mask, or None.
        use_cache: Whether a cache was requested back.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor | None
    past_key_values: KVCacheProto | None
    attention_mask: torch.Tensor | None
    use_cache: bool


class FakeCacheOut(CacheCarryingOutProto):
    """A forward output carrying both a loss and a key-value cache.

    Attributes:
        cache: The per-layer key and value pairs to report.
        loss_value: The loss to report.
    """

    def __init__(
        self,
        *,
        cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
        loss_value: float,
    ) -> None:
        """Hold the values this output reports.

        Args:
            cache: Per-layer key and value pairs.
            loss_value: The loss to report.
        """
        self.cache = cache
        self.loss_value = loss_value

    @property
    def loss(self) -> torch.Tensor:
        """Return the configured loss.

        Returns:
            A scalar tensor.
        """
        return torch.tensor(self.loss_value)

    @property
    def past_key_values(self) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        """Return the configured cache.

        Returns:
            Per-layer key and value pairs.
        """
        return self.cache


class FakeCacheCapableModel(FakeModel):
    """A fake model that reports a key-value cache of a chosen shape.

    Lets the geometry-discovery path and the refusals around it be driven
    without constructing a transformer, while the strategy's own integration
    tests use a real one. What is faked here is the SHAPE a model reports;
    what a real model does with a prefix is not faked anywhere.

    Attributes:
        calls: Every call this model received, so a test can assert what was
            actually passed rather than that something was.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        key_dims: int = 4,
    ) -> None:
        """Configure the cache shape this model reports.

        Args:
            num_layers: Layers to report. Zero reports an empty cache, which
                is the "cannot host a prefix" case.
            num_kv_heads: Key-value heads per layer.
            head_dim: Width of one head's vectors.
            key_dims: Dimensionality of the reported key tensors. Four is the
                real layout; anything else drives the malformed-cache refusal.
        """
        super().__init__(name="cache-capable")
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._key_dims = key_dims
        self.calls: list[RecordedCall] = []

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return one parameter, which is where the probe reads its device.

        Returns:
            A single named tensor on the CPU.
        """
        return [("weight", torch.zeros(1))]

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: KVCacheProto | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> CacheCarryingOutProto:
        """Record the call and report the configured cache.

        Args:
            input_ids: Token ids.
            labels: Targets, or None.
            past_key_values: Prefix cache, or None.
            attention_mask: Mask, or None.
            use_cache: Whether a cache was asked for.

        Returns:
            An output carrying the configured cache shape.
        """
        self.calls.append(
            RecordedCall(
                input_ids=input_ids,
                labels=labels,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
        )
        shape = (1, self._num_kv_heads, 1, self._head_dim)[: self._key_dims]
        cache = [(torch.zeros(shape), torch.zeros(shape)) for _ in range(self._num_layers)]
        return FakeCacheOut(cache=cache, loss_value=1.0)
