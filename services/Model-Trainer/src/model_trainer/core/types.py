from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch


class ConfigLike(Protocol):
    """Protocol for model configuration objects."""


class ParameterLike(Protocol):
    """Protocol for model parameters (tensors with gradients)."""


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
    @classmethod
    def from_pretrained(cls: type[LMModelProto], path: str) -> LMModelProto: ...
    def train(self: LMModelProto) -> None: ...
    def eval(self: LMModelProto) -> None: ...
    def forward(
        self: LMModelProto, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto: ...
    def parameters(self: LMModelProto) -> Sequence[ParameterLike]: ...
    def named_parameters(self: LMModelProto) -> Sequence[tuple[str, NamedParameter]]: ...
    def to(self: LMModelProto, device: str) -> LMModelProto: ...
    def save_pretrained(self: LMModelProto, out_dir: str) -> None: ...

    @property
    def config(self: LMModelProto) -> ConfigLike: ...
