"""The differentiable public surface: ordered matmul and addmm.

The ``__import__`` accessors exist for the reason :mod:`autograd`'s
docstring gives: naming a ``Function`` subclass in an expression trips the
contains-Any check, so the classes are reached dynamically and typed at the
boundary -- the established pattern.
"""

from __future__ import annotations

from typing import Protocol

import torch


class _ApplyMatmulProto(Protocol):
    """``OrderedMatmul.apply``, with the type its stub loses."""

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...


class _ApplyAddmmProto(Protocol):
    """``OrderedAddmm.apply``, with the type its stub loses."""

    def __call__(self, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...


class _MatmulFunctionProto(Protocol):
    """The class object carrying the ordered matmul's ``apply``."""

    apply: _ApplyMatmulProto


class _AddmmFunctionProto(Protocol):
    """The class object carrying the ordered addmm's ``apply``."""

    apply: _ApplyAddmmProto


def _matmul_apply() -> _ApplyMatmulProto:
    """Reach ``OrderedMatmul.apply`` without naming the class in an expression."""
    module = __import__("ordered_kernels.autograd", fromlist=["OrderedMatmul"])
    function: _MatmulFunctionProto = module.OrderedMatmul
    return function.apply


def _addmm_apply() -> _ApplyAddmmProto:
    """Reach ``OrderedAddmm.apply``, typed. See :func:`_matmul_apply`."""
    module = __import__("ordered_kernels.autograd", fromlist=["OrderedAddmm"])
    function: _AddmmFunctionProto = module.OrderedAddmm
    return function.apply


def ordered_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ w`` with both directions' orders owned, differentiably.

    Args:
        x: ``[N, K]``, float32, CUDA.
        w: ``[K, M]``, float32, CUDA.

    Returns:
        ``[N, M]``.
    """
    return _matmul_apply()(x, w)


def ordered_addmm(bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Compute ``bias + x @ w`` with every reduction owned, differentiably.

    Args:
        bias: ``[M]``, float32, CUDA.
        x: ``[N, K]``, float32, CUDA.
        w: ``[K, M]``, float32, CUDA.

    Returns:
        ``[N, M]``.
    """
    return _addmm_apply()(bias, x, w)


__all__ = ["ordered_addmm", "ordered_matmul"]
