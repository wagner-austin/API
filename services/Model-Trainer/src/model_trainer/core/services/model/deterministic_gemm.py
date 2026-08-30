"""Three ways to compute one GEMM, differing only in who chooses the order.

WHY THIS EXISTS. Every determinism control shipped so far -- ``CUBLAS_WORKSPACE_CONFIG``,
``use_deterministic_algorithms``, TF32 off, ``CUBLASLT_WORKSPACE_SIZE=0``, the
SDPA math pin -- constrains WHICH vendor kernel cuBLAS picks. None of them
removes the vendor's remaining freedom, and that freedom is per-architecture
by design: tile sizes and split counts come from heuristics keyed on compute
capability, SM count and L2 size. A different split is a different summation
order, and float addition is not associative, so the same operands round
differently. That is the whole residual the v24 four-card trace is left with,
and no further flag can reach it -- the trace that found it already had every
flag on.

So the next control is not a flag. It is owning the reduction order, which
means not calling the vendor's GEMM. These are the three arms worth measuring:

* :data:`CUBLAS_ARM` is ``addmm``, the baseline the ladder and the trace have
  always run. It is here so the other two are read against it in the same
  record rather than against a number recovered from a different run.

* :data:`FP64_ARM` widens to float64, multiplies, and narrows back. It is
  what a practitioner reaches for first and it is NOT a determinism argument:
  cuBLAS still picks the kernel and still picks the order, so two cards can
  still disagree -- just ~2**-29 further down, where narrowing to float32
  usually but not always rounds the disagreement away. "Usually" is the
  reason to measure it rather than recommend it. Its cost also varies by an
  order of magnitude across the cards under test, since fp64 throughput is
  1:2 on the V100, A100 and A30 and 1:64 on the L40S.

* :data:`RANK1_ARM` is the one with an argument behind it. It accumulates K
  rank-one outer products in ascending k, so the sum is a SEQUENCE OF
  ELEMENTWISE OPERATIONS and there is no reduction for the hardware to
  reorder: every output element is produced by one thread doing one multiply
  and one add, K times, in an order the program fixes rather than the
  heuristic. Vectorisation and occupancy still differ per card and cannot
  change that -- they change which thread runs when, not what it computes.
  Volta and Ampere both implement IEEE-754 binary32 arithmetic, so the same
  operations in the same order give the same bits.

  It is quadratically wasteful of bandwidth -- K passes over an MxN
  accumulator instead of one -- so it is an existence proof and a measurement
  instrument, NOT a proposal for the forward pass. What it establishes, if it
  holds, is that the ceiling is reachable, which is the thing no amount of
  configuration could tell us.

WHY NOT TRITON, WHICH IS THE OBVIOUS ANSWER AND IS ALREADY IN THE IMAGE. A
Triton kernel with a fixed ``BLOCK_K`` and no split-K would keep tensor cores
and most of the speed. It is the right SECOND move and the wrong first one:
``tl.dot`` lowers to the MMA instruction of the target architecture, whose
shape differs between sm_70 and sm_80, so the intra-tile order may move with
the card -- which is the very thing under test, reintroduced one level down.
Establishing that a fixed order suffices has to come first, and it has to
come from an arm where "the order is fixed" is a claim about the program
rather than about a compiler's lowering. This module is also imported on
Windows by the test suite, where Triton does not install at all.
"""

from __future__ import annotations

from typing import Final

import torch

#: ``addmm``: the vendor picks the kernel and the order.
CUBLAS_ARM = "cublas"

#: float64 throughout, narrowed at the end. The vendor still picks the order.
FP64_ARM = "fp64"

#: K rank-one updates in ascending k. The program fixes the order.
RANK1_ARM = "rank1"

#: Every arm, in the order a report should read them: baseline, the cheap
#: attempt, the one with a proof. Declared as a tuple rather than inferred
#: from the dispatch below, so a report can name the arms without importing
#: torch -- the reason :mod:`gemm_shapes` is separate from :mod:`gemm_probe`.
KERNEL_ARMS: Final[tuple[str, ...]] = (CUBLAS_ARM, FP64_ARM, RANK1_ARM)


def cublas_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``bias + x @ w`` on the cuBLASLt path.

    ``addmm`` rather than ``mm``: the fused bias epilogue is what routes this
    to cuBLASLt, measured under ``CUBLASLT_LOG_LEVEL=4``. See
    :mod:`gemm_shapes` for the operand orientation and why it matters.

    Args:
        bias: ``[M]``.
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``.
    """
    return torch.addmm(bias, x, w)


def fp64_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute the same product in float64 and narrow the result.

    Narrowed rather than returned wide, because the question is whether two
    cards produce the same FLOAT32 tensor. Returning float64 would compare a
    quantity the forward pass never holds and would make the arm incomparable
    with the other two.

    Args:
        bias: ``[M]``, float32.
        x: ``[N, K]``, float32.
        w: ``[K, M]``, float32.

    Returns:
        ``[N, M]``, float32.
    """
    wide = torch.addmm(bias.double(), x.double(), w.double())
    return wide.float()


def rank1_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute the same product as K rank-one updates in ascending k.

    The accumulator starts at zero and the bias is added at the END, not
    folded in first. Both are fixed orders and they give different bits: a
    bias added first participates in the rounding of all K subsequent adds.
    Last matches how ``addmm`` is written -- ``bias + (x @ w)`` -- so the arms
    differ in the reduction under study and not also in where the bias went.

    ``addr_`` rather than ``add_(torch.outer(...))``: one elementwise kernel
    over the accumulator instead of an allocation and two, which at K up to
    8192 is the difference between seconds and tens of seconds. It computes
    the same thing.

    Args:
        bias: ``[M]``.
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``.
    """
    accumulator = torch.zeros(x.shape[0], w.shape[1], dtype=x.dtype, device=x.device)
    for k in range(w.shape[0]):
        accumulator.addr_(x[:, k], w[k, :])
    return bias + accumulator


def require_kernel_arm(raw: str) -> str:
    """Return the arm named, refusing anything else.

    Refused rather than defaulted for the reason
    :func:`~model_trainer.core.services.model.control_arms.require_control_arm`
    refuses: a record whose arm was guessed is a record naming a condition it
    may not have run under, and here the arm decides the arithmetic itself.

    Args:
        raw: The flag's value.

    Returns:
        ``raw``, once it is known to be one of :data:`KERNEL_ARMS`.

    Raises:
        ValueError: When it is not.
    """
    if raw not in KERNEL_ARMS:
        raise ValueError(f"kernel must be one of {', '.join(KERNEL_ARMS)}; got {raw!r}")
    return raw


def gemm_by_arm(arm: str, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Dispatch one GEMM to the named arm.

    Args:
        arm: One of :data:`KERNEL_ARMS`.
        bias: ``[M]``.
        x: ``[N, K]``.
        w: ``[K, M]``.

    Returns:
        ``[N, M]``, float32.

    Raises:
        ValueError: Propagated from :func:`require_kernel_arm`.
    """
    named = require_kernel_arm(arm)
    if named == CUBLAS_ARM:
        return cublas_addmm(bias, x, w)
    if named == FP64_ARM:
        return fp64_addmm(bias, x, w)
    return rank1_addmm(bias, x, w)


__all__ = [
    "CUBLAS_ARM",
    "FP64_ARM",
    "KERNEL_ARMS",
    "RANK1_ARM",
    "cublas_addmm",
    "fp64_addmm",
    "gemm_by_arm",
    "rank1_addmm",
    "require_kernel_arm",
]
