"""The autograd Functions over the ordered kernels, quarantined for the stubs.

The same split, for the same reason, as Model-Trainer's ``owned_backward``:
``torch.autograd.Function`` carries ``Any`` in its stub, so any EXPRESSION
naming a subclass trips the contains-Any check, while a class DEFINITION
does not. Definitions live here; :mod:`api` reaches ``apply`` through
``__import__``.

The gradients are the owned arm's gradients, computed by the fast kernels:
``grad_x = grad_out @ Wᵀ`` and ``grad_w = xᵀ @ grad_out`` are each one more
:func:`~ordered_kernels.kernels.gemm` (the transposed views pay one
deterministic contiguous copy each), and the bias gradient is
:func:`~ordered_kernels.kernels.rowsum`. Single backward only.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ordered_kernels.kernels import gemm, gemm_batched, lastdim_sum, rowsum


class SavedTensorsProto(Protocol):
    """The two members of a Function context this module touches."""

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        """Stash tensors the backward will need."""
        ...

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]:
        """The stashed tensors, in the order stashed."""
        ...


class OrderedMatmul(torch.autograd.Function):
    """``x @ w`` with forward and backward reductions program-ordered, fast."""

    @staticmethod
    def forward(ctx: SavedTensorsProto, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute the product by the tiled ascending-k kernel."""
        ctx.save_for_backward(x, w)
        return gemm(x, w, None)

    @staticmethod
    def backward(
        ctx: SavedTensorsProto, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both gradient products through the same kernel, unconditionally."""
        x, w = ctx.saved_tensors
        return gemm(grad_out, w.t(), None), gemm(x.t(), grad_out, None)


class OrderedAddmm(torch.autograd.Function):
    """``bias + x @ w`` with every reduction owned, the bias gradient's too."""

    @staticmethod
    def forward(
        ctx: SavedTensorsProto, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Product in ascending k, bias added last, in one kernel."""
        ctx.save_for_backward(x, w)
        return gemm(x, w, bias)

    @staticmethod
    def backward(
        ctx: SavedTensorsProto, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Row-accumulated bias gradient; kernel GEMMs for the other two."""
        x, w = ctx.saved_tensors
        return rowsum(grad_out), gemm(grad_out, w.t(), None), gemm(x.t(), grad_out, None)


class OrderedBatchedMatmul(torch.autograd.Function):
    """``x[b] @ w[b]`` per slice, both directions' reductions owned.

    The batched twin of :class:`OrderedMatmul`, for attention's two
    products: ``grad_x[b] = grad_out[b] @ w[b]ᵀ`` and
    ``grad_w[b] = x[b]ᵀ @ grad_out[b]`` are each one more
    :func:`~ordered_kernels.kernels.gemm_batched` on transposed views,
    which pay one deterministic contiguous copy each inside the kernel.
    Single backward only, like its 2-D twin.
    """

    @staticmethod
    def forward(ctx: SavedTensorsProto, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute every slice's product by the tiled ascending-k kernel."""
        ctx.save_for_backward(x, w)
        return gemm_batched(x, w)

    @staticmethod
    def backward(
        ctx: SavedTensorsProto, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both gradient products through the batched kernel, unconditionally."""
        x, w = ctx.saved_tensors
        return (
            gemm_batched(grad_out, w.transpose(-1, -2)),
            gemm_batched(x.transpose(-1, -2), grad_out),
        )


class OrderedSoftmax(torch.autograd.Function):
    """Last-dim softmax with BOTH of its row reductions owned.

    The forward's denominator and the backward's projection are the only
    reductions softmax holds, and each goes through
    :func:`~ordered_kernels.kernels.lastdim_sum`: the backward is the
    closed-form ``p * (g - Σⱼ gⱼpⱼ)`` with its row sum computed in ascending
    column order. The row max is taken with torch -- an fp32 max is exact in
    any order -- and is a constant shift the derivative does not see, which
    is why the backward needs no saved scores, only the probabilities.
    Everything else is elementwise. Single backward only.
    """

    @staticmethod
    def forward(ctx: SavedTensorsProto, scores: torch.Tensor) -> torch.Tensor:
        """Each row's ``exp(x - max)`` over the owned ascending-order sum."""
        row_max = scores.amax(dim=-1, keepdim=True)
        exps = torch.exp(scores - row_max)
        probs = exps / lastdim_sum(exps).unsqueeze(-1)
        ctx.save_for_backward(probs)
        return probs

    @staticmethod
    def backward(ctx: SavedTensorsProto, grad_out: torch.Tensor) -> torch.Tensor:
        """``p * (g - Σ gp)``, the row reduction in ascending order."""
        (probs,) = ctx.saved_tensors
        projected = lastdim_sum(grad_out * probs).unsqueeze(-1)
        return probs * (grad_out - projected)


__all__ = [
    "OrderedAddmm",
    "OrderedBatchedMatmul",
    "OrderedMatmul",
    "OrderedSoftmax",
    "SavedTensorsProto",
]
