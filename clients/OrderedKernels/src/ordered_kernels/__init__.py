"""Fast fixed-order GEMMs: bit-identical across GPUs, at tiled-kernel speed.

The production-shaped successor to the rank-one instrument: the same
program-owned ascending-k order (and the same two-rounding multiply-add, so
the seven-GPU record corpus is a bit-exact oracle), implemented as a
shared-memory tiled CUDA kernel instead of K passes over an accumulator.
See :mod:`ordered_kernels.kernels` for the order contract.
"""

from ordered_kernels.api import ordered_addmm, ordered_matmul
from ordered_kernels.attention import ordered_causal_attention, ordered_softmax
from ordered_kernels.kernels import gemm, gemm_batched, lastdim_sum, rowsum
from ordered_kernels.modules import (
    OrderedConv1D,
    OrderedLinear,
    OrderedSdpaAttention,
    use_ordered_attention,
    use_ordered_kernels,
)

__all__ = [
    "OrderedConv1D",
    "OrderedLinear",
    "OrderedSdpaAttention",
    "gemm",
    "gemm_batched",
    "lastdim_sum",
    "ordered_addmm",
    "ordered_causal_attention",
    "ordered_matmul",
    "ordered_softmax",
    "rowsum",
    "use_ordered_attention",
    "use_ordered_kernels",
]
