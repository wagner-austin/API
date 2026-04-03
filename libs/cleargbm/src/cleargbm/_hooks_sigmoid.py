"""Sigmoid backend hooks for cleargbm.

Scalar and vectorized sigmoid functions. Tests inject fakes, production
uses real implementations.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SigmoidBackend(Protocol):
    """Protocol for scalar sigmoid backend."""

    def __call__(self, x: float) -> float:
        """Compute sigmoid function.

        Args:
            x: Input value (log-odds).

        Returns:
            Probability in [0, 1].
        """
        ...


class SigmoidArrayBackend(Protocol):
    """Protocol for vectorized sigmoid backend."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute sigmoid function for array.

        Args:
            x: Input array (log-odds).

        Returns:
            Probabilities in [0, 1].
        """
        ...


def _default_sigmoid(x: float) -> float:
    """Python scalar sigmoid implementation.

    Clips input to [-500, 500] to prevent overflow.

    Args:
        x: Input value (log-odds).

    Returns:
        Probability in [0, 1].
    """
    x_clipped = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x_clipped))


def _default_sigmoid_array(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Python vectorized sigmoid implementation.

    Uses numpy for efficient array operations. Clips to [-500, 500].

    Args:
        x: Input array (log-odds).

    Returns:
        Probabilities in [0, 1].
    """
    x_clipped: NDArray[np.float64] = np.clip(x, -500.0, 500.0)
    result: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-x_clipped))
    return result


# Module-level hooks for sigmoid backend.
# Production sets these to Rust implementations at startup.
# Tests override to provide Python fakes.
_sigmoid_backend: SigmoidBackend = _default_sigmoid
_sigmoid_array_backend: SigmoidArrayBackend = _default_sigmoid_array


def sigmoid(x: float) -> float:
    """Compute sigmoid function.

    Delegates to the active backend hook.

    Args:
        x: Input value (log-odds).

    Returns:
        Probability in [0, 1].
    """
    return _sigmoid_backend(x)


def sigmoid_array(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute sigmoid function for array.

    Delegates to the active backend hook.

    Args:
        x: Input array (log-odds).

    Returns:
        Probabilities in [0, 1].
    """
    return _sigmoid_array_backend(x)


__all__ = [
    "SigmoidArrayBackend",
    "SigmoidBackend",
    "sigmoid",
    "sigmoid_array",
]
