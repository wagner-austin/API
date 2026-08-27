from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Protocol

import torch


class ConfigLike(Protocol):
    """Protocol for model configuration objects."""


class TorchStateValue(Protocol):
    """Protocol for one value inside a torch-serialized state payload.

    Deliberately memberless: model, optimizer and RNG state entries are
    heterogeneous (tensors, numbers, nested containers) and are
    round-tripped through ``torch.save``/``torch.load`` without being
    introspected. A consumer that needs a concrete type narrows with
    ``isinstance``.
    """


class ParameterLike(Protocol):
    """Protocol for model parameters (tensors with gradients)."""

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the parameter tensor."""
        ...

    def numel(self) -> int:
        """Return total number of elements in the parameter tensor."""
        ...


class LoadStateDictResultProto(Protocol):
    """Protocol for the result of a module ``load_state_dict`` call.

    Torch returns a named tuple of missing and unexpected keys; callers
    here never inspect it, so the protocol carries no members.
    """


class OptimizerProto(Protocol):
    """Protocol for PyTorch optimizer instances."""

    def zero_grad(self, *, set_to_none: bool = ...) -> None: ...
    def step(self) -> None: ...
    def state_dict(self) -> dict[str, TorchStateValue]: ...
    def load_state_dict(self, state_dict: dict[str, TorchStateValue]) -> None: ...


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

    def state_dict(self: LMModelProto) -> dict[str, torch.Tensor]:
        """Return the model's parameter and buffer tensors by name."""
        ...

    def load_state_dict(
        self: LMModelProto, state_dict: dict[str, torch.Tensor]
    ) -> LoadStateDictResultProto:
        """Load parameter and buffer tensors by name."""
        ...

    def gradient_checkpointing_enable(self: LMModelProto) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        ...

    @property
    def config(self: LMModelProto) -> ConfigLike:
        """Return model configuration."""
        ...


class HookValue(Protocol):
    """Protocol for one value a torch hook is handed.

    Deliberately memberless, for the reason :class:`TorchStateValue` is: what
    reaches a forward hook is heterogeneous by construction -- a tensor from
    most modules, a tuple from a transformer block, a ``ModelOutput`` mapping
    from the model itself, and ``None`` or a bool among a module's positional
    arguments. A consumer that needs a concrete type narrows with
    ``isinstance`` or with ``torch.is_tensor``.
    """


class HookHandleProto(Protocol):
    """Protocol for the handle torch returns when a hook is registered."""

    def remove(self) -> None:
        """Detach the hook this handle was returned for."""
        ...


class ForwardHookProto(Protocol):
    """Protocol for a callable torch invokes after a module computes.

    Positional-only, because torch calls hooks positionally. Returning None
    is what tells torch to keep the module's own output; a hook here observes
    and never substitutes.
    """

    def __call__(
        self,
        module: TracedModuleProto,
        args: tuple[HookValue, ...],
        output: HookValue,
        /,
    ) -> None:
        """Observe one module's inputs and output."""
        ...


class ForwardPreHookProto(Protocol):
    """Protocol for a callable torch invokes before a module computes.

    Returning None is what tells torch to keep the module's own inputs.
    """

    def __call__(self, module: TracedModuleProto, args: tuple[HookValue, ...], /) -> None:
        """Observe one module's positional inputs."""
        ...


class TracedModuleProto(Protocol):
    """Protocol for the torch module surface a forward trace needs.

    Narrow on purpose. :class:`LMModelProto` describes what a language model
    can be asked to DO; this describes the module graph underneath it, which
    only a trace looks at. Folding these four methods into ``LMModelProto``
    would oblige every fake language model in the suite to grow a module
    graph it has no use for.

    ``named_modules`` and ``children`` are declared as generators because
    that is what torch returns. Declaring a ``Sequence`` would typecheck and
    then fail at runtime for anyone who indexed the result.
    """

    def named_modules(self) -> Generator[tuple[str, TracedModuleProto], None, None]:
        """Yield every module in the graph with its dotted path, root first."""
        ...

    def children(self) -> Generator[TracedModuleProto, None, None]:
        """Yield this module's immediate children."""
        ...

    def register_forward_hook(self, hook: ForwardHookProto, /) -> HookHandleProto:
        """Call ``hook`` after this module computes, until the handle is removed."""
        ...

    def register_forward_pre_hook(self, hook: ForwardPreHookProto, /) -> HookHandleProto:
        """Call ``hook`` before this module computes, until the handle is removed."""
        ...


class TracedLMModelProto(LMModelProto, TracedModuleProto, Protocol):
    """A language model whose module graph can also be traced.

    What the GPT-2 constructor actually returns. Every caller that only wants
    to train, score or save is served by :class:`LMModelProto` and is unaffected
    by this being the declared return type, since this is a subtype of it.

    ``to`` is redeclared only to narrow the return type: the inherited one
    returns :class:`LMModelProto`, which would lose the module graph the moment
    a caller moved the model to a device.
    """

    def to(self: TracedLMModelProto, device: str) -> TracedLMModelProto:
        """Move the model to a device and return it, still traceable."""
        ...
