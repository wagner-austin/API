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
"""

from __future__ import annotations

import hashlib
import struct

import torch

from model_trainer.core.services.model.gemm_shapes import GEMM_SEED, GemmShape

#: How many bytes of the digest fold into the observation. Six bytes is 48
#: bits, comfortably inside float64's exact-integer range (2**53); taking
#: seven would start rounding and two different tensors could then record the
#: same number, which is the one failure this observation must not have.
DIGEST_BYTES = 6


def _fold_digest(payload: bytes) -> float:
    """Reduce bytes to a float that changes whenever they do.

    Args:
        payload: The bytes to identify.

    Returns:
        The first :data:`DIGEST_BYTES` of the SHA-256, as an exact float.
    """
    digest = hashlib.sha256(payload).digest()
    return float(int.from_bytes(digest[:DIGEST_BYTES], "big"))


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


def _output_bytes(output: torch.Tensor) -> bytes:
    """Render a float32 tensor as its raw bytes.

    Goes through a Python list rather than ``.numpy()``, which returns an
    untyped array this package's mypy settings reject. The round trip is
    exact: widening float32 to float64 loses nothing, and packing a value
    that came from a float32 back into one is the identity. (A NaN payload
    is the exception, and a NaN here would be a finding rather than a
    measurement.)

    Args:
        output: The tensor to render, on the CPU.

    Returns:
        Its float32 bytes, little-endian.
    """
    values: list[float] = output.contiguous().flatten().tolist()
    return struct.pack(f"<{len(values)}f", *values)


def require_reproduced(
    first: torch.Tensor, second: torch.Tensor, shape: GemmShape, device: str
) -> torch.Tensor:
    """Return one of two runs of a call, refusing if they differ.

    A separate function rather than a branch inside :func:`gemm_identity`,
    because it is the one part of this module a test can exercise in both
    directions. A CPU GEMM is deterministic, so the failing arm cannot be
    reached by running one -- and an arm no test can reach is an arm nobody
    has checked says what it means.

    Args:
        first: The first run's output.
        second: The second run's.
        shape: The call, for the message.
        device: Where it ran, for the message.

    Returns:
        ``first``, once the two are known to agree.

    Raises:
        RuntimeError: If they differ. Within one device and one process this
            must be exact; if it is not, nothing measured ACROSS cards means
            anything, so the run stops rather than recording a number whose
            own device cannot reproduce it.
    """
    if not torch.equal(first, second):
        raise RuntimeError(
            f"a GEMM M{shape['rows']}xK{shape['inner']}xN{shape['cols']} did not "
            f"reproduce itself on {device}; nothing measured across cards would mean anything"
        )
    return first


def describe_output(output: torch.Tensor) -> tuple[float, float]:
    """Reduce an output tensor to its identity and its magnitude.

    Args:
        output: The GEMM result, on the CPU.

    Returns:
        ``(folded digest, float64 sum)``. Summed in float64 on the CPU so the
        reduction cannot itself differ between devices -- the only
        device-dependent step must be the matmul.
    """
    return _fold_digest(_output_bytes(output)), float(output.double().sum().item())


def gemm_identity(shape: GemmShape, device: str) -> tuple[float, float]:
    """Run one GEMM twice and describe the result.

    Args:
        shape: The call to measure.
        device: Device to run it on.

    Returns:
        ``(folded digest, float64 sum)`` of the output.

    Raises:
        RuntimeError: Propagated from :func:`require_reproduced` when the same
            call on the same device produced two different tensors.
    """
    first = gemm_output(shape, device).cpu()
    second = gemm_output(shape, device).cpu()
    return describe_output(require_reproduced(first, second, shape, device))


__all__ = [
    "DIGEST_BYTES",
    "describe_output",
    "gemm_identity",
    "gemm_operands",
    "gemm_output",
    "require_reproduced",
]
