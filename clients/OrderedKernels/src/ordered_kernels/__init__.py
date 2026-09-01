"""Fast fixed-order GEMMs: bit-identical across GPUs, at tiled-kernel speed.

The production-shaped successor to the rank-one instrument: the same
program-owned ascending-k order (and the same two-rounding multiply-add, so
the seven-GPU record corpus is a bit-exact oracle), implemented as a
shared-memory tiled CUDA kernel instead of K passes over an accumulator.
See :mod:`ordered_kernels.kernels` for the order contract.
"""

from ordered_kernels.api import ordered_addmm, ordered_matmul
from ordered_kernels.kernels import gemm, rowsum
from ordered_kernels.modules import OrderedConv1D, OrderedLinear, use_ordered_kernels

__all__ = [
    "OrderedConv1D",
    "OrderedLinear",
    "gemm",
    "ordered_addmm",
    "ordered_matmul",
    "rowsum",
    "use_ordered_kernels",
]
