"""Run one training step and describe every gradient and updated weight.

:mod:`train_step_plan` says what this measures and what each result is
called; this runs one. The model, the input and the seed come from
:func:`~known_answer_probe.probe_model_and_input` -- the same builder every
forward measurement uses -- so a train-step record and a forward-trace record
describe the same model, and a difference between them is the backward pass
and nothing else.

WHAT ONE STEP IS HERE. A forward pass with labels, ``loss.backward()``, and
one in-place SGD update of every parameter at a fixed step size. Digested:
every parameter's gradient, then its post-update value, in
``named_parameters`` order, plus the loss. Not digested: optimizer moments,
because there is no optimizer -- the update is the one arithmetic an
optimizer adds that touches every parameter, and it is elementwise, so
adding AdamW's state would measure more memory for the same class of
operation.

THE MODEL STAYS IN EVAL MODE, DELIBERATELY. ``probe_model_and_input``
returns it that way, and switching to train mode would enable dropout, which
multiplies every activation by a Philox-derived mask. Whether two cards draw
identical masks is a question about the RNG, not about the arithmetic under
study, and a probe that entangled the two could attribute nothing. Gradients
flow identically in eval mode; what differs is only that no activation is
masked.

WITHIN-CARD DETERMINISM IS ALREADY PINNED BY THE CALLER'S POSTURE.
``probe_determinism`` routes through ``apply_determinism``, which calls
``torch.use_deterministic_algorithms(True)`` -- that is what forces the
embedding gradient's scatter-add onto its sorted deterministic path instead
of atomic adds. The probe still proves it rather than assuming it: the whole
step runs twice from a fresh model and the run refuses to report if the two
disagree, the same discipline :func:`~tensor_digest.require_reproduced`
applies to one GEMM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from typing_extensions import TypedDict

from model_trainer.core.services.model.kernel_arm_modules import use_kernel_arm
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import ProbeShape
from model_trainer.core.services.model.tensor_digest import describe_tensor
from model_trainer.core.services.model.train_step_plan import (
    GRAD_KIND,
    TRAIN_STEP_LR,
    UPDATED_KIND,
)
from model_trainer.core.types import NamedParameter


class TrainTensor(TypedDict):
    """One digested tensor of one training step.

    Attributes:
        kind: :data:`~train_step_plan.GRAD_KIND` or
            :data:`~train_step_plan.UPDATED_KIND`.
        path: The parameter's dotted path.
        digest: The folded digest of the tensor's bytes.
        total: Its chunked float64 sum.
    """

    kind: str
    path: str
    digest: float
    total: float


class SteppedModelProto(Protocol):
    """The surface :func:`digest_step_tensors` needs from a stepped model.

    Its own Protocol rather than :class:`~model_trainer.core.types.LMModelProto`
    because the function reads exactly one thing -- the walked parameters --
    and taking the full model protocol would put ``save_pretrained`` and
    ``gradient_checkpointing_enable`` into the signature of a function that
    walks a list.
    """

    def named_parameters(self) -> Sequence[tuple[str, NamedParameter]]:
        """Return the parameters to digest, in walk order."""
        ...


def digest_step_tensors(model: SteppedModelProto) -> tuple[TrainTensor, ...]:
    """Digest every gradient, apply the SGD update, digest every new value.

    Per parameter, in ``named_parameters`` order: the gradient's digest, then
    the update, then the updated value's digest. Interleaved rather than two
    passes so the walk order in the record is the walk order that ran, and so
    a parameter without a gradient is refused before anything is mutated.

    The update writes through ``param.detach()``: a view of the same storage
    that does not require grad, so the in-place ``add_`` neither trips
    autograd's leaf-mutation check nor needs a ``no_grad`` block around a
    walk that otherwise only reads.

    Args:
        model: The model whose backward pass has already run.

    Returns:
        Two entries per parameter, gradient first.

    Raises:
        ValueError: When a parameter has no gradient. A missing gradient
            means the parameter did not participate in the loss, and a record
            that silently skipped it would claim a step it did not take.
        ValueError: Propagated from :func:`~tensor_digest.describe_tensor`
            for a NaN gradient, which in a probe is a finding, not a number.
    """
    tensors: list[TrainTensor] = []
    for path, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            raise ValueError(
                f"{path} has no gradient after backward; a step that skipped it "
                "would be recorded as one the model did not take"
            )
        digest, total = describe_tensor(grad)
        tensors.append(TrainTensor(kind=GRAD_KIND, path=path, digest=digest, total=total))

        updated = param.detach()
        updated.add_(grad, alpha=-TRAIN_STEP_LR)
        digest, total = describe_tensor(updated)
        tensors.append(TrainTensor(kind=UPDATED_KIND, path=path, digest=digest, total=total))
    return tuple(tensors)


def train_step_once(
    device: str, shape: ProbeShape, *, kernel: str
) -> tuple[
    tuple[TrainTensor, ...],
    float,
]:
    """Build one rung's model fresh and take one step on it.

    Args:
        device: Device to run on.
        shape: The rung to build.
        kernel: Which arithmetic the model's matmuls use, by
            :data:`~deterministic_gemm.KERNEL_ARMS` name. Applied before the
            forward pass, so a treated arm's backward runs through autograd's
            derivatives of the treated operations -- which fixes the FORWARD
            order only. The backward reductions stay the vendor's under every
            arm; owning those too would need custom backward kernels, and a
            record must not be read as claiming otherwise.

    Returns:
        ``(digested tensors, the loss)``.

    Raises:
        ValueError: Propagated from
            :func:`~known_answer_probe.probe_model_and_input` for a shape
            whose sequence exceeds its vocabulary, from
            :func:`~kernel_arm_modules.use_kernel_arm` for an unknown arm, or
            from :func:`digest_step_tensors`.
    """
    model, ids = probe_model_and_input(device, shape)
    use_kernel_arm(model, kernel)
    outputs = model.forward(input_ids=ids, labels=ids)
    loss = outputs.loss
    # `torch.autograd.backward` rather than the method, which torch's stubs
    # leave untyped -- the same spelling `train_cost` uses for the same
    # reason. They run identical arithmetic.
    torch.autograd.backward([loss])
    return digest_step_tensors(model), float(loss.item())


def require_step_reproduced(
    first: tuple[TrainTensor, ...],
    second: tuple[TrainTensor, ...],
    first_loss: float,
    second_loss: float,
    device: str,
) -> tuple[tuple[TrainTensor, ...], float]:
    """Return the first step's results, refusing if the second differed.

    The step-level twin of :func:`~tensor_digest.require_reproduced`, and
    separate from :func:`train_step_identity` for the same reason that one is
    separate: the arithmetic is deterministic within a device, so the failing
    arm cannot be reached by running it, and an arm no test can reach is an
    arm nobody has confirmed says what it means.

    Digests are compared rather than tensors because the second run's model
    must be buildable after the first is freed -- ``xl`` gradients beside two
    live models is more memory than a V100 has -- and two tensors whose
    48-bit folded digests and chunked sums both match are byte-identical to
    the same standard every cross-card comparison here uses.

    Args:
        first: The first run's digested tensors.
        second: The second run's.
        first_loss: The first run's loss.
        second_loss: The second run's.
        device: Where they ran, for the message.

    Returns:
        ``(first, first_loss)``, once the runs are known to agree.

    Raises:
        RuntimeError: If they differ.
    """
    if first != second or first_loss != second_loss:
        raise RuntimeError(
            f"a train step did not reproduce itself on {device}; "
            "nothing measured across cards would mean anything"
        )
    return first, first_loss


def train_step_identity(
    device: str, shape: ProbeShape, *, kernel: str
) -> tuple[
    tuple[TrainTensor, ...],
    float,
]:
    """Take the same step twice from a fresh model and describe the result.

    Args:
        device: Device to run on.
        shape: The rung to run.
        kernel: One of :data:`~deterministic_gemm.KERNEL_ARMS`.

    Returns:
        ``(digested tensors, the loss)``.

    Raises:
        RuntimeError: Propagated from :func:`require_step_reproduced` when
            the same step on the same device produced two different results.
        ValueError: Propagated from :func:`train_step_once`.
    """
    first, first_loss = train_step_once(device, shape, kernel=kernel)
    second, second_loss = train_step_once(device, shape, kernel=kernel)
    return require_step_reproduced(first, second, first_loss, second_loss, device)


__all__ = [
    "SteppedModelProto",
    "TrainTensor",
    "digest_step_tensors",
    "require_step_reproduced",
    "train_step_identity",
    "train_step_once",
]
