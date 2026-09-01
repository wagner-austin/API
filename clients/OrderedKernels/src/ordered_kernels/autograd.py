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

from ordered_kernels.kernels import gemm, rowsum


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


__all__ = ["OrderedAddmm", "OrderedMatmul", "SavedTensorsProto"]
