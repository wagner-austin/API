"""Time one attention call under one backend, and say what it cost in memory.

The price of the correctness result in :mod:`sdpa_probe`: pinning
``SDPBackend.MATH`` makes attention bit-identical across a V100, an A30 and
an A100, and this measures what that costs PER CALL. What a whole forward
pass costs is :mod:`forward_cost`, and the two share
:mod:`timing_harness` precisely so that the two numbers can be read against
each other.

WHY MEMORY IS MEASURED AND NOT ONLY TIME. The math path materialises the full
``[batch, heads, seq, seq]`` score matrix; the fused kernel does not. That is
a QUALITATIVE difference, not a constant factor, and a table of seconds would
report it as "slower" right up to the point where it stops fitting on the
card at all. Peak allocation is recorded beside every timing, and an
out-of-memory is recorded as a result rather than raised.
"""

from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend

from model_trainer.core.services.model.gemm_timing import (
    BATCHES,
    INNER,
    WARMUP,
    synchroniser,
)
from model_trainer.core.services.model.sdpa_probe import sdpa_output
from model_trainer.core.services.model.sdpa_shapes import SDPA_SEED, SdpaCostShape
from model_trainer.core.services.model.timing_harness import (
    MeasuredCost,
    backend_context,
    time_calls,
    timed_or_unfitted,
)


def cost_operands(
    shape: SdpaCostShape, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one timed call's query, key and value.

    Laid out the way ``_split_heads`` lays them out -- a view then a permute,
    leaving them non-contiguous -- because that is what the model hands the
    dispatcher and strides reach kernel selection. Generated on the device
    rather than on the CPU and moved, unlike
    :func:`~sdpa_probe.sdpa_operands`: this measures TIME, the values do not
    enter the answer, and a 4096-token batch would otherwise be built in host
    memory and copied for nothing.

    Args:
        shape: The call to build for.
        device: Where to build.

    Returns:
        ``(query, key, value)``, each ``[batch, heads, sequence_len,
        head_dim]``.
    """
    torch.manual_seed(SDPA_SEED)
    flat = [
        torch.randn(
            shape["batch"],
            shape["sequence_len"],
            shape["heads"] * shape["head_dim"],
            dtype=torch.float32,
            device=device,
        )
        for _ in range(3)
    ]
    return (
        flat[0]
        .view(shape["batch"], shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
        flat[1]
        .view(shape["batch"], shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
        flat[2]
        .view(shape["batch"], shape["sequence_len"], shape["heads"], shape["head_dim"])
        .permute(0, 2, 1, 3),
    )


def measure_sdpa(shape: SdpaCostShape, device: str, backend: SDPBackend | None) -> MeasuredCost:
    """Time one attention call and read its peak allocation.

    Args:
        shape: The call to time.
        device: Device to run it on.
        backend: The backend to force, or None for the dispatcher's choice.

    Returns:
        The cost.

    Raises:
        torch.cuda.OutOfMemoryError: When the device cannot hold the call.
            Caught by :func:`~timing_harness.timed_or_unfitted`.
        RuntimeError: When the forced backend has no kernel for this call.
            Not caught: this benchmark prices a backend the selection probe
            already showed to run on every card, so a refusal would mean it
            is pricing something else and must be loud.
    """
    query, key, value = cost_operands(shape, device)

    def run() -> None:
        sdpa_output(query, key, value)

    with backend_context(backend):
        return time_calls(run, synchroniser(device), device, WARMUP, INNER, BATCHES)


def time_sdpa(shape: SdpaCostShape, device: str, backend: SDPBackend | None) -> MeasuredCost | None:
    """Measure seconds per call and peak memory for one attention call.

    Args:
        shape: The call to time.
        device: Device to run it on.
        backend: The backend to force, or None for the dispatcher's choice.

    Returns:
        The cost, or None when the call did not fit in device memory.

    Raises:
        RuntimeError: Propagated for a failure that is neither an
            out-of-memory nor a no-kernel refusal.
    """

    def run() -> MeasuredCost:
        return measure_sdpa(shape, device, backend)

    return timed_or_unfitted(run)


__all__ = [
    "cost_operands",
    "measure_sdpa",
    "time_sdpa",
]
