"""Run one GEMM and describe its output tensor exactly.

:mod:`gemm_shapes` says which calls exist and why; this issues one and reduces
its result to two numbers a run record can carry.

WHY A FOLDED DIGEST RATHER THAN A SUM. The question is bitwise identity across
cards, and a sum can hide a difference: two elements differing by ``+d`` and
``-d`` cancel exactly. A hash of the output bytes cannot cancel. Forty-eight
bits of it fold into a float64 without loss -- every integer below 2**53 is
exactly representable -- so bitwise identity becomes float equality, which is
what :func:`~platform_core.run_record.agree_across_runs` already tests. The
sum is recorded beside it to say how LARGE a difference is once the digest
says there is one; alone it would be the weaker check, and alone it is not
used.

THE FOLD ITSELF LIVES IN :mod:`tensor_digest`, which the forward trace also
uses. A second spelling of it here would let the two drift, and a digest
recorded by one could then not be read against a digest recorded by the
other. What does NOT come from there is the SUM: this module sums the whole
output in float64 in one torch call, while the trace sums in fixed chunks
because its tensors are far larger. Adopting the chunked sum here would
change every gemm sum already recorded, so it keeps its own.
"""

from __future__ import annotations

import torch

from model_trainer.core.services.model.gemm_shapes import GEMM_SEED, GemmShape
from model_trainer.core.services.model.tensor_digest import (
    DIGEST_BYTES,
    describe_tensor,
    require_reproduced,
)


def gemm_operands(shape: GemmShape, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one call's operands, identically on every device.

    Generated on the CPU under a fixed seed and then moved, never generated
    on the device. The CUDA RNG is per-device, so generating there would hand
    two cards different inputs and produce a difference that says nothing
    about how they multiply.

    Args:
        shape: The call to build for.
        device: Where the operands must end up.

    Returns:
        ``(bias, x, w)`` for ``addmm(bias, x, w)``.
    """
    torch.manual_seed(GEMM_SEED)
    bias = torch.randn(shape["rows"], dtype=torch.float32)
    x = torch.randn(shape["cols"], shape["inner"], dtype=torch.float32)
    w = torch.randn(shape["inner"], shape["rows"], dtype=torch.float32)
    return bias.to(device), x.to(device), w.to(device)


def gemm_output(shape: GemmShape, device: str) -> torch.Tensor:
    """Compute one GEMM, on the cuBLASLt path.

    ``addmm`` rather than ``mm``: the fused bias epilogue is what routes this
    to cuBLASLt, and ``mm`` was measured to reach the legacy ``cublasSgemm``
    entry point instead, logging nothing under a trace. A probe on the wrong
    entry point would answer a question nobody asked.

    Args:
        shape: The call to run.
        device: Device to run it on.

    Returns:
        The output tensor, still on ``device``.
    """
    bias, x, w = gemm_operands(shape, device)
    return torch.addmm(bias, x, w)


def gemm_description(shape: GemmShape) -> str:
    """Name one call, for a self-reproduction failure message.

    Args:
        shape: The call.

    Returns:
        e.g. ``a GEMM M1024xK4096xN64``.
    """
    return f"a GEMM M{shape['rows']}xK{shape['inner']}xN{shape['cols']}"


def gemm_identity(shape: GemmShape, device: str) -> tuple[float, float]:
    """Run one GEMM twice and describe the result.

    Args:
        shape: The call to measure.
        device: Device to run it on.

    Returns:
        ``(folded digest, float64 sum)`` of the output.

    Raises:
        RuntimeError: Propagated from
            :func:`~tensor_digest.require_reproduced` when the same call on
            the same device produced two different tensors.
    """
    first = gemm_output(shape, device).cpu()
    second = gemm_output(shape, device).cpu()
    return describe_tensor(require_reproduced(first, second, gemm_description(shape), device))


__all__ = [
    "DIGEST_BYTES",
    "gemm_description",
    "gemm_identity",
    "gemm_operands",
    "gemm_output",
]
