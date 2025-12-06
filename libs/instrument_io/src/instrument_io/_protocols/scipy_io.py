"""Protocol definitions for scipy.io module.

Provides type-safe interface to scipy.io.loadmat function
without importing scipy directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DtypeProtocol(Protocol):
    """Protocol for numpy dtype."""

    @property
    def name(self) -> str:
        """Return dtype name."""
        ...


PythonValue = str | int | float | bool | None | list["PythonValue"]


class NdarrayProtocol(Protocol):
    """Protocol for numpy.ndarray."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape."""
        ...

    @property
    def dtype(self) -> DtypeProtocol:
        """Return array data type."""
        ...

    def tolist(self) -> PythonValue:
        """Convert array to Python list."""
        ...


# scipy.io.loadmat returns dict with string keys and numpy arrays
MATDict = dict[str, NdarrayProtocol | str | int | float]


class _LoadMatFn(Protocol):
    """Protocol for scipy.io.loadmat function."""

    def __call__(self, file_name: str | Path) -> MATDict:
        """Load MATLAB .mat file.

        Args:
            file_name: Path to .mat file.

        Returns:
            Dictionary mapping variable names to values.
        """
        ...


def _load_mat(path: Path) -> MATDict:
    """Load MATLAB .mat file with proper typing via Protocol.

    Args:
        path: Path to .mat file.

    Returns:
        Dictionary mapping variable names to numpy arrays or other types.
    """
    scipy_io_mod = __import__("scipy.io", fromlist=["loadmat"])
    loadmat_fn: _LoadMatFn = scipy_io_mod.loadmat
    return loadmat_fn(path)


__all__ = [
    "DtypeProtocol",
    "MATDict",
    "NdarrayProtocol",
    "PythonValue",
    "_load_mat",
]
