"""Time a whole training step, with and without the attention pin.

The measurement every previous page in this thread declined to make. The
forward benchmark found pinning ``SDPBackend.MATH`` costs 1.14-1.31x on a
forward pass and no extra memory at all -- and said, each time, that nothing
there licensed a claim about training, because the backward pass through the
math backend is a different computation.

TWO REASONS IT REALLY IS DIFFERENT, both structural rather than guessed:

* The forward ran under ``no_grad``. A training forward keeps what backward
  needs, and what the math path needs is the full ``[batch, heads, seq, seq]``
  score matrix -- the tensor the fused kernel exists to avoid keeping. The
  forward benchmark's finding that the pin costs no extra memory cannot
  survive that, and the point of this module is to find out by how much.
* The model runs in ``train()`` rather than ``eval()``, so dropout is live and
  a nonzero ``dropout_p`` reaches the attention dispatcher. That is a
  different call than the one the forward benchmark timed.

WHAT A STEP IS HERE IS WHAT THE TRAINER DOES, read out of
``base_trainer_loop`` rather than invented: forward, read the loss,
``zero_grad(set_to_none=True)``, ``torch.autograd.backward``, clip the
gradient norm, ``optimizer.step()``. The optimizer is AdamW through the same
``_get_optimizer_for_config`` the trainer calls, so this prices the step that
service actually runs. A benchmark that timed a hand-rolled step would be
pricing something nobody runs.

WHY THE OPTIMIZER IS INCLUDED EVEN THOUGH IT DILUTES THE RATIO. It is a
constant added to both arms, so including it makes the multiplier smaller --
and that is the multiplier a training run actually experiences. Excluding it
to make the attention cost look larger would be choosing the framing that
flatters the finding.
"""

from __future__ import annotations

from typing import Final

import torch
from torch.nn.attention import SDPBackend
from typing_extensions import TypedDict

from model_trainer.core.services.model.forward_cost import (
    GPT2_VOCAB,
    ForwardCostShape,
    forward_model_and_input,
)
from model_trainer.core.services.model.gemm_timing import synchroniser
from model_trainer.core.services.model.timing_harness import (
    MeasuredCost,
    backend_context,
    time_calls,
    timed_or_unfitted,
)
from model_trainer.core.services.training.base_trainer_core import (
    _get_optimizer_for_config,
)
from model_trainer.core.services.training.trainer_grad_utils import (
    _clip_grad_norm_with_return,
)
from model_trainer.core.types import OptimizerProto, TracedLMModelProto

#: Steps discarded before measuring. Three rather than the forward
#: benchmark's two, because AdamW allocates its moment buffers lazily on the
#: FIRST ``step()`` -- so one warmup step would leave that allocation inside
#: the measurement, and it happens once per run rather than once per step.
TRAIN_WARMUP = 3

#: Steps per timed batch. One, for the reason the forward benchmark gives:
#: a training step issues thousands of launches and has amortised them.
TRAIN_INNER = 1

#: Timed batches. The median is reported.
TRAIN_BATCHES = 5

#: Learning rate. Any value trains; none of them changes what a step costs,
#: and a benchmark that pretended to have tuned one would be claiming a
#: relevance it does not have.
TRAIN_LR = 1e-4

#: Gradient-norm ceiling, matching the trainer's own default clipping.
TRAIN_CLIP = 1.0

#: The optimizer name the trainer's config uses, resolved through the
#: trainer's own map so this cannot drift from what the service runs.
TRAIN_OPTIMIZER = "adamw"


class TrainStep(TypedDict):
    """One row's model, input and optimizer, ready to step.

    Attributes:
        model: The model, in TRAIN mode -- dropout live, gradients tracked.
        ids: Input token ids, used as both input and labels.
        optimizer: AdamW over the model's parameters.
    """

    model: TracedLMModelProto
    ids: torch.Tensor
    optimizer: OptimizerProto


#: The rows. Smaller than the forward sweep's on purpose: a training step
#: holds parameters, gradients and two AdamW moments, so gpt2-large needs
#: about 12 GB before a single activation. The batch descends as the model
#: grows so the unpinned arm fits on a 16 GB V100 -- and `large-b1-s1024` is
#: deliberately left near that edge, because a row that does not fit is the
#: strongest cost result available and the sweep should be able to find one.
TRAIN_SHAPES: Final[tuple[ForwardCostShape, ...]] = (
    {
        "name": "small-b8-s512",
        "model_size": "small",
        "batch": 8,
        "sequence_len": 512,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "small-b4-s1024",
        "model_size": "small",
        "batch": 4,
        "sequence_len": 1024,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "medium-b4-s512",
        "model_size": "medium",
        "batch": 4,
        "sequence_len": 512,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "medium-b2-s1024",
        "model_size": "medium",
        "batch": 2,
        "sequence_len": 1024,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "large-b1-s1024",
        "model_size": "large",
        "batch": 1,
        "sequence_len": 1024,
        "vocab_size": GPT2_VOCAB,
    },
    {
        "name": "gate-tiny-b1-s64",
        "model_size": "tiny",
        "batch": 1,
        "sequence_len": 64,
        "vocab_size": 512,
    },
)


def train_step_setup(shape: ForwardCostShape, device: str) -> TrainStep:
    """Build one row's model, input and optimizer, in training posture.

    The model comes from :func:`~forward_cost.forward_model_and_input`, which
    builds through the same constructor everything else here uses, and is then
    put into TRAIN mode -- the forward benchmark left it in eval, and the
    difference is live dropout and tracked gradients.

    Args:
        shape: The row to build.
        device: Where to build it.

    Returns:
        The model, its input and its optimizer.
    """
    model, ids = forward_model_and_input(shape, device)
    model.train()
    optimizer_cls = _get_optimizer_for_config(TRAIN_OPTIMIZER)
    return TrainStep(model=model, ids=ids, optimizer=optimizer_cls(model.parameters(), lr=TRAIN_LR))


def run_train_step(step: TrainStep) -> None:
    """Run one training step, in the shape the trainer runs it.

    Read out of ``base_trainer_loop`` rather than invented, including the
    ``.item()`` read of the loss: that is a host-device sync the real loop
    performs every step, and leaving it out would time a step nobody runs.

    Args:
        step: The row's model, input and optimizer.
    """
    outputs = step["model"].forward(input_ids=step["ids"], labels=step["ids"])
    loss = outputs.loss
    float(loss.item())
    step["optimizer"].zero_grad(set_to_none=True)
    torch.autograd.backward([loss])
    _clip_grad_norm_with_return(step["model"].parameters(), max_norm=TRAIN_CLIP)
    step["optimizer"].step()


def measure_train_step(step: TrainStep, device: str, backend: SDPBackend | None) -> MeasuredCost:
    """Time one training step and read its peak allocation.

    Args:
        step: The row's model, input and optimizer.
        device: Device being timed.
        backend: The attention backend to force, or None for the
            dispatcher's choice.

    Returns:
        The cost.

    Raises:
        torch.cuda.OutOfMemoryError: When the device cannot hold the step.
            Caught by :func:`~timing_harness.timed_or_unfitted`.
    """

    def run() -> None:
        run_train_step(step)

    with backend_context(backend):
        return time_calls(
            run, synchroniser(device), device, TRAIN_WARMUP, TRAIN_INNER, TRAIN_BATCHES
        )


def time_train_step(
    step: TrainStep, device: str, backend: SDPBackend | None
) -> MeasuredCost | None:
    """Measure seconds and peak memory for one training step.

    Args:
        step: The row's model, input and optimizer.
        device: Device being timed.
        backend: The attention backend to force, or None.

    Returns:
        The cost, or None when the step did not fit in device memory.
    """

    def run() -> MeasuredCost:
        return measure_train_step(step, device, backend)

    return timed_or_unfitted(run)


__all__ = [
    "TRAIN_BATCHES",
    "TRAIN_CLIP",
    "TRAIN_INNER",
    "TRAIN_LR",
    "TRAIN_OPTIMIZER",
    "TRAIN_SHAPES",
    "TRAIN_WARMUP",
    "TrainStep",
    "measure_train_step",
    "run_train_step",
    "time_train_step",
    "train_step_setup",
]
