from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch


class ConfigLike(Protocol):
    """Protocol for model configuration objects."""


class ParameterLike(Protocol):
    """Protocol for model parameters (tensors with gradients)."""

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the parameter tensor."""
        ...

    def numel(self) -> int:
        """Return total number of elements in the parameter tensor."""
        ...


class OptimizerProto(Protocol):
    """Protocol for PyTorch optimizer instances."""

    def zero_grad(self, *, set_to_none: bool = ...) -> None: ...
    def step(self) -> None: ...


class OptimizerCtorProto(Protocol):
    """Protocol for PyTorch optimizer constructors (e.g., AdamW class)."""

    def __call__(
        self,
        params: Sequence[ParameterLike],
        *,
        lr: float,
    ) -> OptimizerProto: ...


class ForwardOutProto(Protocol):
    @property
    def loss(self: ForwardOutProto) -> torch.Tensor: ...


class NamedParameter(Protocol):
    """Protocol for named parameter tuples from named_parameters()."""

    @property
    def requires_grad(self) -> bool: ...

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None: ...

    @property
    def grad(self) -> torch.Tensor | None: ...

    def detach(self) -> torch.Tensor: ...

    def clone(self) -> torch.Tensor: ...


class LMModelProto(Protocol):
    """Protocol for language model instances.

    Defines the interface for HuggingFace-compatible language models
    used throughout the training pipeline.
    """

    @classmethod
    def from_pretrained(cls: type[LMModelProto], path: str) -> LMModelProto:
        """Load model from pretrained weights."""
        ...

    def train(self: LMModelProto) -> None:
        """Set model to training mode."""
        ...

    def eval(self: LMModelProto) -> None:
        """Set model to evaluation mode."""
        ...

    def forward(
        self: LMModelProto, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        """Forward pass through the model."""
        ...

    def parameters(self: LMModelProto) -> Sequence[ParameterLike]:
        """Return model parameters."""
        ...

    def named_parameters(
        self: LMModelProto,
    ) -> Sequence[tuple[str, NamedParameter]]:
        """Return named parameters."""
        ...

    def to(self: LMModelProto, device: str) -> LMModelProto:
        """Move model to device."""
        ...

    def save_pretrained(self: LMModelProto, out_dir: str) -> None:
        """Save model to directory."""
        ...

    def gradient_checkpointing_enable(self: LMModelProto) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        ...

    @property
    def config(self: LMModelProto) -> ConfigLike:
        """Return model configuration."""
        ...
