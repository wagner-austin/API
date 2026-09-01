"""The autograd Functions behind the owned arm, quarantined for the stubs.

These belong conceptually in :mod:`deterministic_gemm`, beside the arm they
implement, and they are NOT there for one reason: ``torch.autograd.Function``
carries ``Any`` in its stub, so any EXPRESSION naming a subclass -- including
``_OwnedMatmul.apply`` -- trips this package's contains-Any check. A class
DEFINITION does not. So the classes live here, where only definitions name
them, and :mod:`deterministic_gemm` reaches them through ``__import__`` --
the same dance :mod:`kernel_arm_modules` does for ``transformers``' Conv1D,
for the same reason.

The arithmetic itself is :func:`~deterministic_gemm.rank1_matmul` and
:func:`~deterministic_gemm.accumulate_rows`; see :data:`~deterministic_gemm.OWNED_ARM`
for what the arm claims and what it deliberately leaves to autograd.
"""

from __future__ import annotations

from typing import Protocol

import torch

from model_trainer.core.services.model.deterministic_gemm import (
    accumulate_rows,
    rank1_matmul,
)


class SavedTensorsProto(Protocol):
    """The two members of a Function context this module touches.

    Torch's ``FunctionCtx`` stub does not declare ``saved_tensors`` -- it is
    materialized onto the backward context at runtime -- so annotating the
    ctx with the stub type would leave every read of it untyped. A Protocol
    naming exactly what is used keeps the surface stated, the same reason
    ``apply_determinism`` takes leaf objects.
    """

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        """Stash tensors the backward will need."""
        ...

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]:
        """The stashed tensors, in the order stashed."""
        ...


class OwnedMatmul(torch.autograd.Function):
    """``x @ w`` with the forward AND backward reductions program-ordered.

    Single backward only: the gradients this returns are plain tensors, so
    differentiating through them a second time is not supported -- and the
    train-step probe takes one step, which is one backward.
    """

    @staticmethod
    def forward(ctx: SavedTensorsProto, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute the product by ascending-k rank-one updates."""
        ctx.save_for_backward(x, w)
        return rank1_matmul(x, w)

    @staticmethod
    def backward(
        ctx: SavedTensorsProto, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both gradient products by the same fixed order.

        ``grad_x = grad_out @ w.T`` reduces over the output width and
        ``grad_w = x.T @ grad_out`` over the batch; each is just a matmul,
        so each goes through ``rank1_matmul``. Both are computed
        unconditionally rather than gated on ``needs_input_grad`` -- an
        unneeded gradient is discarded by autograd, and a branch here would
        be an arm the probe's models never drive.
        """
        x, w = ctx.saved_tensors
        return rank1_matmul(grad_out, w.t()), rank1_matmul(x.t(), grad_out)


class OwnedAddmm(torch.autograd.Function):
    """``bias + x @ w`` with every reduction owned, the bias gradient's too.

    The bias must live INSIDE the function: written as ``bias + owned
    product``, the outer add would be autograd's, and its bias gradient --
    ``grad_out.sum(dim=0)`` -- would be a vendor-ordered reduction over the
    batch, un-owning exactly one gradient per projection.
    """

    @staticmethod
    def forward(
        ctx: SavedTensorsProto, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Compute the product in ascending k, then add the bias last."""
        ctx.save_for_backward(x, w)
        return bias + rank1_matmul(x, w)

    @staticmethod
    def backward(
        ctx: SavedTensorsProto, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Row-accumulate the bias gradient; rank-one the other two."""
        x, w = ctx.saved_tensors
        return (
            accumulate_rows(grad_out),
            rank1_matmul(grad_out, w.t()),
            rank1_matmul(x.t(), grad_out),
        )


__all__ = ["OwnedAddmm", "OwnedMatmul", "SavedTensorsProto"]
