"""Package-level injection seams for covenant_ml.

Strict typing only: no Any, no casts, no stubs.

``cuda_runtime_available`` is bound to the real answer this package can give,
so a caller needs nothing wired first. Tests narrow it through
``covenant_ml.testing.set_cuda_hook`` and restore it with ``reset_cuda_hook``.
"""

from __future__ import annotations

from typing import Protocol


class CudaRuntimeAvailableProtocol(Protocol):
    """Protocol for the run-time CUDA availability answer."""

    def __call__(self) -> bool:
        """Report whether a CUDA device may be used.

        Returns:
            True when nothing rules CUDA out at run time.
        """
        ...


def _real_cuda_runtime_available() -> bool:
    """Report whether anything rules CUDA out at run time.

    Nothing does, here. This package's only CUDA signal is XGBoost's build
    info, which _detect_cuda_available reads and which this answer is combined
    with; probing for a present device would need a dependency covenant_ml
    does not carry, and _resolve_device already raises when CUDA is asked for
    and the build does not support it. The seam exists so a caller that does
    know better -- a test on a CI box without a GPU, say -- can say no.

    Returns:
        True.
    """
    return True


cuda_runtime_available: CudaRuntimeAvailableProtocol = _real_cuda_runtime_available


__all__ = [
    "CudaRuntimeAvailableProtocol",
    "cuda_runtime_available",
]
