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

from model_trainer.core.services.model.deterministic_gemm import gemm_by_arm
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


def gemm_output(shape: GemmShape, device: str, *, kernel: str) -> torch.Tensor:
    """Compute one GEMM under one kernel arm.

    The arm is keyword-only and required, with no default, for the reason
    ``probe_determinism``'s controls are: a caller that did not say which
    arithmetic it wanted is a caller whose record cannot say either. On
    :data:`~deterministic_gemm.CUBLAS_ARM` this is ``addmm`` rather than
    ``mm``, because the fused bias epilogue is what routes it to cuBLASLt and
    ``mm`` was measured to reach the legacy ``cublasSgemm`` entry point
    instead, logging nothing under a trace.

    Args:
        shape: The call to run.
        device: Device to run it on.
        kernel: One of :data:`~deterministic_gemm.KERNEL_ARMS`.

    Returns:
        The output tensor, still on ``device``.

    Raises:
        ValueError: Propagated from
            :func:`~deterministic_gemm.require_kernel_arm` for an unknown arm.
    """
    bias, x, w = gemm_operands(shape, device)
    return gemm_by_arm(kernel, bias, x, w)


def gemm_description(shape: GemmShape, kernel: str) -> str:
    """Name one call, for a self-reproduction failure message.

    The arm is in the message because all three produce a tensor of the same
    shape: a failure reading only ``M1024xK4096xN64`` would not say which
    arithmetic failed to reproduce itself.

    Args:
        shape: The call.
        kernel: The arm it ran under.

    Returns:
        e.g. ``a rank1 GEMM M1024xK4096xN64``.
    """
    return f"a {kernel} GEMM M{shape['rows']}xK{shape['inner']}xN{shape['cols']}"


def gemm_identity(shape: GemmShape, device: str, *, kernel: str) -> tuple[float, float]:
    """Run one GEMM twice under one arm and describe the result.

    Args:
        shape: The call to measure.
        device: Device to run it on.
        kernel: One of :data:`~deterministic_gemm.KERNEL_ARMS`.

    Returns:
        ``(folded digest, float64 sum)`` of the output.

    Raises:
        RuntimeError: Propagated from
            :func:`~tensor_digest.require_reproduced` when the same call on
            the same device produced two different tensors.
        ValueError: Propagated from
            :func:`~deterministic_gemm.require_kernel_arm` for an unknown arm.
    """
    first = gemm_output(shape, device, kernel=kernel).cpu()
    second = gemm_output(shape, device, kernel=kernel).cpu()
    described = require_reproduced(first, second, gemm_description(shape, kernel), device)
    return describe_tensor(described)


__all__ = [
    "DIGEST_BYTES",
    "gemm_description",
    "gemm_identity",
    "gemm_operands",
    "gemm_output",
]
