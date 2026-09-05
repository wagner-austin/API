"""Causal attention with every reduction program-ordered.

WHY THIS EXISTS. The GTX 1630 (sm_75) broke the fixed-order scoring identity
on exactly three of 150 items -- all of them, and only them, at 15- and
16-token sequences, below anything the probe tables sample -- while every
operation the arms OWN agreed between the cards bit for bit. Attention was
the leading suspect because it is the arithmetic no module swap reaches:
under the math pin its matmuls and its softmax run whatever kernels the
vendor selects for the shape. This module removes that freedom.

WHAT IS OWNED, AND WHAT IS DELIBERATELY NOT. Attention holds exactly three
reductions and they are all owned: the ``q @ k^T`` products and the
``probs @ v`` products go through the batched fixed-order GEMM, and the
softmax denominator goes through the fixed-order row sum. The row MAX is
taken with torch: an fp32 max is exact whatever order it is computed in, so
there is no rounding freedom to own. Everything else -- the scale multiply,
the mask add, the subtract, ``exp``, the divide -- is elementwise, computed
by torch: an elementwise op has no reduction order, and the stage digests
measure whether that trust holds rather than assume it.

DIFFERENTIABLE, SAME BITS FORWARD. The reductions run inside the
:mod:`autograd` Functions, so gradients flow at any length with the
backward's reductions owned too: each matmul's two gradient products are
batched fixed-order GEMMs, and softmax's backward projection is the owned
row sum. The forward arithmetic inside those Functions is the kernel
composition this module always ran -- the pinned inference records are the
regression gate on that claim.

WHAT THIS IS NOT. It is not a bit-reproduction of torch's math SDPA -- the
whole point is DIFFERENT arithmetic for the same function. Correctness is
asserted numerically against the math backend in the suite; cross-card
bit-identity is what the records establish.
"""

from __future__ import annotations

import math

import torch

from ordered_kernels.api import ordered_batched_matmul, ordered_row_softmax

#: The additive causal term: finite scores keep their value, future
#: positions become -inf, whose exp is exactly 0.0 in every rounding mode.
CAUSAL_FILL: float = float("-inf")


def causal_bias(length: int, device: torch.device) -> torch.Tensor:
    """The additive causal mask for one ``[length, length]`` score matrix.

    Args:
        length: Sequence length.
        device: Where the scores live.

    Returns:
        ``[length, length]`` float32: 0.0 at and below the diagonal,
        ``-inf`` strictly above it.
    """
    zeros = torch.zeros(length, length, dtype=torch.float32, device=device)
    return zeros.masked_fill(
        torch.ones(length, length, dtype=torch.bool, device=device).triu(diagonal=1),
        CAUSAL_FILL,
    )


def ordered_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Last-dim softmax whose denominator is summed in ascending order.

    Differentiable: the forward arithmetic is unchanged from the original
    kernel composition -- the same amax, the same exp, the same owned sum,
    the same divide -- but it now runs inside a ``Function`` whose backward
    owns softmax's other row reduction too. The stage digests and the
    pinned inference records are the proof the routing moved no forward
    bit, not this sentence.

    Args:
        scores: ``[R, C]``, float32, CUDA. Rows holding ``-inf`` entries are
            fine -- ``exp`` maps them to exactly 0.0 -- but every row must
            hold at least one finite entry, which a causal row always does
            (its diagonal).

    Returns:
        ``[R, C]``: each row's ``exp(x - max)`` divided by the owned sum.
    """
    return ordered_row_softmax(scores)


def ordered_causal_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Causal scaled-dot-product attention, every reduction owned.

    Args:
        query: ``[B, H, L, D]``, float32, CUDA; strided views are fine and
            are copied contiguous inside the batched GEMM.
        key: Same shape.
        value: Same shape.

    Returns:
        ``[B, H, L, D]``.

    Raises:
        ValueError: For mismatched shapes, or propagated from the kernels'
            operand checks.
    """
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            f"q, k and v must share one shape; got {tuple(query.shape)}, "
            f"{tuple(key.shape)}, {tuple(value.shape)}"
        )
    if query.dim() != 4:
        raise ValueError(f"expected [batch, heads, length, dim], got {query.dim()}-D")
    batch, heads, length, dim = (int(s) for s in query.shape)
    folded = batch * heads
    scores = ordered_batched_matmul(
        query.reshape(folded, length, dim), key.transpose(-1, -2).reshape(folded, dim, length)
    )
    # For GPT-2's head dim of 64 the scale is 0.125 -- a power of two, so
    # the multiply shifts exponents without rounding and its placement
    # relative to the matmul cannot move a bit.
    scores = scores * (1.0 / math.sqrt(float(dim)))
    scores = scores + causal_bias(length, query.device)
    probs = ordered_softmax(scores.reshape(folded * length, length))
    out = ordered_batched_matmul(
        probs.reshape(folded, length, length), value.reshape(folded, length, dim)
    )
    return out.reshape(batch, heads, length, dim)


__all__ = ["CAUSAL_FILL", "causal_bias", "ordered_causal_attention", "ordered_softmax"]
