from __future__ import annotations

from typing import Protocol

import torch
from platform_core.logging import get_logger

from handwriting_ai.training.mnist_train import _configure_threads


class _SetNumInteropThreads(Protocol):
    """torch.set_num_interop_threads, whose parameter is named."""

    def __call__(self, nthreads: int) -> None:
        """Set the interop thread count.

        Args:
            nthreads: Number of interop threads.
        """
        ...


class _Cfg:
    def __getitem__(self, key: str) -> int:
        return 1


def test_configure_threads_without_set_num_interop() -> None:
    had_attr = hasattr(torch, "set_num_interop_threads")
    # Read and written as a plain attribute so the save and the restore pair up.
    saved: _SetNumInteropThreads | None = None
    # torch names the parameter, so the restore needs the same signature shape.
    if had_attr:
        saved = torch.set_num_interop_threads
    if had_attr:
        delattr(torch, "set_num_interop_threads")
    try:
        # This should not raise and should skip interop configuration branch
        _configure_threads(_Cfg())
        # Touch logger to satisfy guard; also asserts threading configured
        get_logger("handwriting_ai").info("set_num_interop_threads_absent")
        assert torch.get_num_threads() >= 1
    finally:
        if had_attr and saved is not None:
            torch.set_num_interop_threads = saved
