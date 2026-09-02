"""Typed accessors over the two tensor methods torch's stubs leave as Any.

``Tensor.split`` and pair-indexed ``__getitem__`` come back untyped from the
stubs, and under this workspace's settings one Any poisons every expression
downstream of it. Same remedy as the cupy surface in :mod:`kernels`: reach
the member dynamically and type it at the boundary with a Protocol, once,
here -- never inline at call sites, where the dance would be repeated and
drift.
"""

from __future__ import annotations

from typing import Protocol

import torch


class _SplitProto(Protocol):
    """``Tensor.split`` with the type its stub loses."""

    def __call__(self, split_size: int, dim: int) -> tuple[torch.Tensor, ...]: ...


def split_three(
    tensor: torch.Tensor, width: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed QKV projection into its exact three slices.

    Args:
        tensor: The packed tensor.
        width: Width of each slice along ``dim``.
        dim: The dimension to split.

    Returns:
        The three slices, in order.

    Raises:
        ValueError: When the split does not yield exactly three -- a QKV
            width mismatch, refused rather than silently misassigned.
    """
    splitter: _SplitProto = tensor.split
    parts = splitter(width, dim)
    if len(parts) != 3:
        raise ValueError(f"expected three slices of width {width}, got {len(parts)}")
    return parts[0], parts[1], parts[2]


def head_slice(tensor: torch.Tensor, batch: int, head: int) -> torch.Tensor:
    """One ``[length, dim]`` head slice of a ``[B, H, L, D]`` tensor.

    A named accessor rather than inline ``tensor[batch, head]`` because the
    stub types multi-index ``__getitem__`` loosely enough that the workspace
    settings flag the expression; the annotation here is the boundary.

    Args:
        tensor: The batched-heads tensor.
        batch: Batch index.
        head: Head index.

    Returns:
        ``tensor[batch, head]``, typed.
    """
    sliced: torch.Tensor = tensor[batch, head]
    return sliced


__all__ = ["head_slice", "split_three"]
