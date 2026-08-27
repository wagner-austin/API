"""Run one probe rung with every tensor crossing a module boundary digested.

The attribution step for a MODEL, where :mod:`gemm_probe` is the attribution
step for a matmul. A rung reports one loss; by the time two cards' losses
differ, thousands of kernels have run and the difference has been summed into
a single scalar that says nothing about which one carried it. This records a
digest of every tensor that enters or leaves a module, in execution order, so
the first differing digest names an operation.

WHAT IS HOOKED, AND WHY THAT LEAVES NO GAP. Every module's OUTPUT, and every
LEAF module's INPUT. The pairing matters: an operation that is not a module
-- a residual add, a scale, ``scaled_dot_product_attention`` -- can only sit
between one module's output and the next module's input, and a leaf's input is
where such a tensor is caught. Concretely, ``attn.c_proj``'s input IS the
attention operation's output, which is not otherwise observable, because
attention in transformers 4.46 is a function call inside ``GPT2SdpaAttention``
rather than a submodule of it.

The root module is deliberately NOT hooked. A forward hook fires from
``__call__``, and the probe calls ``forward`` directly, so a hook there could
never fire -- registering one would leave a hook in the record's shape that
does nothing, which reads as coverage that is not there.

THE HOOKS DO NOT PERTURB WHAT THEY MEASURE, and the record proves rather than
asserts it: the loss is recorded alongside the tensors, and it must equal what
the untraced ladder reported for the same rung on the same card under the same
condition. If it does not, the instrument changed the arithmetic and nothing
else in the record can be read.
"""

from __future__ import annotations

import torch
from typing_extensions import TypedDict

from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import ProbeShape
from model_trainer.core.services.model.tensor_digest import describe_tensor
from model_trainer.core.services.model.trace_plan import INPUT_KIND, OUTPUT_KIND
from model_trainer.core.types import (
    ForwardHookProto,
    ForwardPreHookProto,
    HookHandleProto,
    HookValue,
    TracedModuleProto,
)


class TracedTensor(TypedDict):
    """One tensor that crossed a module boundary.

    Attributes:
        step: The hook call that recorded it, counting from zero in execution
            order. Assigned per CALL rather than per tensor, so a module
            returning three tensors contributes one step and three indices.
        path: The module's dotted path.
        module_class: The class of the module, e.g. ``Conv1D``.
        kind: :data:`~trace_plan.INPUT_KIND` or
            :data:`~trace_plan.OUTPUT_KIND`.
        index: Which tensor of that call.
        digest: The folded digest of its bytes.
        total: Its chunked float64 sum.
    """

    step: int
    path: str
    module_class: str
    kind: str
    index: int
    digest: float
    total: float


def tensors_in(value: HookValue) -> tuple[torch.Tensor, ...]:
    """Find the tensors a module handed to or received from a hook.

    Args:
        value: A hook's ``output``, or one element of its ``args``.

    Returns:
        The tensors in it: the value itself when it is one, its tensor
        elements when it is a tuple, and none otherwise. "Otherwise" is a real
        case rather than a defensive one -- ``GPT2Model`` returns a
        ``ModelOutput``, which is a mapping -- and it contributes nothing on
        purpose: every tensor inside such a result is also the output of some
        module further in, and is recorded there under a name that says which.
    """
    if torch.is_tensor(value):
        return (value,)
    if isinstance(value, tuple):
        # Two typing details, both forced by this package's settings rather
        # than by taste. The tuple is bound to a typed name before iterating,
        # because narrowing with `isinstance(..., tuple)` gives
        # `tuple[Any, ...]` and expressions of type Any are forbidden here.
        # And the element test is `torch.is_tensor`, a TypeGuard, rather than
        # `isinstance(item, torch.Tensor)`: naming the class in an expression
        # is itself an Any-typed expression under torch's stubs.
        items: tuple[HookValue, ...] = value
        return tuple(item for item in items if torch.is_tensor(item))
    return ()


class ForwardTrace:
    """Collects what the hooks see, in the order they see it.

    A small mutable object rather than a closure over a list, because the
    execution counter and the collected tensors have to move together and
    every hook needs both.
    """

    def __init__(self) -> None:
        """Start an empty trace at step zero."""
        self.step = 0
        self.tensors: list[TracedTensor] = []

    def record(
        self,
        module: TracedModuleProto,
        path: str,
        kind: str,
        values: tuple[torch.Tensor, ...],
    ) -> None:
        """Describe every tensor of one hook call.

        The step advances even when the call carried no tensor, so a step
        number keeps meaning "the nth hook call" rather than "the nth call
        that happened to carry something" -- which would renumber everything
        downstream if a module's output shape ever changed.

        Args:
            module: The module the hook fired for.
            path: Its dotted path.
            kind: Whether these are its inputs or its output.
            values: The tensors to describe.

        Raises:
            ValueError: Propagated from
                :func:`~tensor_digest.describe_tensor` for a NaN or a dtype
                it cannot render exactly.
        """
        step = self.step
        self.step += 1
        module_class = type(module).__name__
        for index, tensor in enumerate(values):
            digest, total = describe_tensor(tensor)
            self.tensors.append(
                TracedTensor(
                    step=step,
                    path=path,
                    module_class=module_class,
                    kind=kind,
                    index=index,
                    digest=digest,
                    total=total,
                )
            )


def _output_hook(trace: ForwardTrace, path: str) -> ForwardHookProto:
    """Build the hook that records one module's output.

    Args:
        trace: Where to record.
        path: The module's dotted path.

    Returns:
        The hook.
    """

    def hook(module: TracedModuleProto, args: tuple[HookValue, ...], output: HookValue, /) -> None:
        trace.record(module, path, OUTPUT_KIND, tensors_in(output))

    return hook


def _input_hook(trace: ForwardTrace, path: str) -> ForwardPreHookProto:
    """Build the hook that records one module's positional inputs.

    Args:
        trace: Where to record.
        path: The module's dotted path.

    Returns:
        The hook.
    """

    def hook(module: TracedModuleProto, args: tuple[HookValue, ...], /) -> None:
        found: list[torch.Tensor] = []
        for item in args:
            found.extend(tensors_in(item))
        trace.record(module, path, INPUT_KIND, tuple(found))

    return hook


def install_hooks(model: TracedModuleProto, trace: ForwardTrace) -> tuple[HookHandleProto, ...]:
    """Hook every module's output and every leaf module's input.

    Args:
        model: The model to instrument.
        trace: Where the hooks record.

    Returns:
        Every handle registered, for the caller to remove. Returned rather
        than stored, because a trace that left its hooks installed would
        silently instrument whatever the process ran next.
    """
    handles: list[HookHandleProto] = []
    for name, module in model.named_modules():
        if name == "":
            # The root. See the module docstring: its hook could not fire.
            continue
        handles.append(module.register_forward_hook(_output_hook(trace, name)))
        if next(module.children(), None) is None:
            handles.append(module.register_forward_pre_hook(_input_hook(trace, name)))
    return tuple(handles)


def traced_forward(device: str, shape: ProbeShape) -> tuple[tuple[TracedTensor, ...], float]:
    """Run one rung with every module boundary digested.

    Args:
        device: Device to run on.
        shape: The rung to run.

    Returns:
        ``(traced tensors in execution order, the reported loss)``. The loss
        is the control described in the module docstring, not a second
        measurement.

    Raises:
        ValueError: Propagated from
            :func:`~known_answer_probe.probe_model_and_input` for a shape
            whose sequence exceeds its vocabulary, or from
            :func:`~tensor_digest.describe_tensor` for a tensor it cannot
            render exactly.
    """
    model, ids = probe_model_and_input(device, shape)
    trace = ForwardTrace()
    handles = install_hooks(model, trace)
    try:
        with torch.no_grad():
            outputs = model.forward(input_ids=ids, labels=ids)
        loss = float(outputs.loss.item())
    finally:
        for handle in handles:
            handle.remove()
    return tuple(trace.tensors), loss


__all__ = [
    "ForwardTrace",
    "TracedTensor",
    "install_hooks",
    "tensors_in",
    "traced_forward",
]
