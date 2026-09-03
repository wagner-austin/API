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
    """Protocol for model parameters (tensors with gradients).

    ``grad`` and ``detach`` are declared because the docstring's parenthetical
    is the whole point of the type: a protocol that could describe a parameter
    but not read its gradient or take a value snapshot of it cannot express
    "this step changed the weights", which is the only way to tell a training
    step from a forward pass that returns a loss.
    """

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the parameter tensor."""
        ...

    @property
    def grad(self) -> torch.Tensor | None:
        """Return the accumulated gradient, or None before any backward."""
        ...

    def numel(self) -> int:
        """Return total number of elements in the parameter tensor."""
        ...

    def detach(self) -> torch.Tensor:
        """Return the same storage, off the autograd graph."""
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

    ``from_pretrained`` IS NOT DECLARED HERE, AND ITS ABSENCE IS THE POINT.
    A ``PeftModel`` does not satisfy the one-argument form: its
    ``from_pretrained`` is ``(cls, model, model_id, ...)``, two required
    parameters, because an adapter is a delta and the base model it applies
    to has to be supplied again. Declaring the one-argument form here made
    every PEFT model structurally claim a classmethod it does not have, and
    the claim was believed at exactly one call site -- the best-checkpoint
    restore -- which crashed at runtime after a whole training run had been
    spent. Loading a class from a path is a property of the CLASS, not of a
    model instance, so it belongs on a loader protocol beside the concrete
    class that actually offers it. See ``_HFModelClassProto`` for the
    HuggingFace form and ``_GPT2Loader`` for the GPT-2 one.
    """

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

    def set_submodule(self, target: str, module: torch.nn.Module) -> None:
        """Replace the submodule at a dotted path.

        Added 2026-08-30 for the kernel arms, which run a rung with its
        matmul-bearing modules swapped for ones that fix the reduction order.
        ``torch.nn.Module.set_submodule`` rather than ``setattr`` on a parent:
        it takes the same dotted path ``named_modules`` already yields, so a
        swap needs no second walk to find the parent and no attribute name
        assembled by hand -- and it is a typed method rather than a dynamic
        attribute write, which is what lets this stay inside a Protocol.
        """
        ...

    def register_forward_hook(self, hook: ForwardHookProto, /) -> HookHandleProto:
        """Call ``hook`` after this module computes, until the handle is removed."""
        ...

    def register_forward_pre_hook(self, hook: ForwardPreHookProto, /) -> HookHandleProto:
        """Call ``hook`` before this module computes, until the handle is removed."""
        ...

    def get_submodule(self, target: str) -> TracedModuleProto:
        """Return the submodule at a dotted path.

        The addressed counterpart of ``named_modules``, which walks. A caller
        that already knows the path it wants should not have to scan the graph
        to reach it.

        Args:
            target: Dotted path, e.g. ``transformer.h.17.mlp.c_proj``.

        Returns:
            The submodule, itself traceable.

        Raises:
            AttributeError: If no submodule exists at that path. Torch's
                message names an attribute rather than the path a caller
                supplied, so callers that resolve a configured path establish
                membership against ``named_modules`` first and raise their own
                error. See ``model_trainer.core.services.model.editing.sites``.
        """
        ...

    def get_parameter(self, target: str) -> EditableParameterProto:
        """Return the parameter at a dotted path, writable.

        The write access is on the PARAMETER type rather than on a second
        model protocol, which keeps the boundary where the danger is: a caller
        holding a model can read every parameter through ``named_parameters``
        and can only change one by naming it here.

        Args:
            target: Dotted path, e.g.
                ``transformer.h.17.mlp.c_proj.weight``.

        Returns:
            The parameter, writable through :class:`EditableParameterProto`.

        Raises:
            AttributeError: If no parameter exists at that path, with the same
                caveat and the same remedy as ``get_submodule``.
        """
        ...

    @property
    def training(self) -> bool:
        """Whether the module is in training mode.

        Declared here rather than on :class:`LMModelProto`, which can already
        CALL ``train()`` and ``eval()`` but has no way to read back which one
        took effect. It belongs to the module surface, and putting it there
        spares every fake language model in the suite a flag it never sets.
        """
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


class EditableParameterProto(Protocol):
    """The parameter surface an in-place weight edit needs.

    Separate from :class:`ParameterLike` and :class:`NamedParameter`, which
    describe a parameter that can be READ. This one can also be written, and
    the distinction is the safety property: only code that asked for this type
    can change a model's weights, and every fake in the suite that merely
    reports a shape stays unaffected.

    ``copy_`` rather than ``add_`` is the single mutator, deliberately. An
    edit computes the whole new matrix and then installs it, so applying an
    update and restoring a snapshot are the same operation with different
    arguments, and there is no second code path whose arithmetic could drift
    from the first.
    """

    @property
    def shape(self: EditableParameterProto) -> torch.Size:
        """Return the shape of the parameter tensor."""
        ...

    def detach(self: EditableParameterProto) -> torch.Tensor:
        """Return the same storage, off the autograd graph."""
        ...

    def copy_(self: EditableParameterProto, src: torch.Tensor) -> torch.Tensor:
        """Overwrite this parameter's storage with ``src``.

        Args:
            src: Tensor to copy in. Must match this parameter's shape.

        Returns:
            This parameter's tensor, now holding ``src``'s values.
        """
        ...
