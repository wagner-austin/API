"""Protocol definitions for numpy types without importing numpy.

These Protocols allow type-safe interaction with numpy arrays returned
by external libraries (rainbow, pyteomics) without requiring numpy
as a direct dependency for type checking.
"""

from __future__ import annotations

from typing import Protocol


class DTypeProtocol(Protocol):
    """Protocol for numpy.dtype.

    Provides access to dtype name for type inspection.
    """

    @property
    def name(self) -> str:
        """Return the dtype name (e.g., 'float64', 'int32')."""
        ...


class NdArrayProtocol(Protocol):
    """Protocol for numpy.ndarray without importing numpy.

    Supports the subset of ndarray interface needed for instrument data:
    - Shape inspection
    - Conversion to Python lists
    - Length and indexing
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array dimensions."""
        ...

    @property
    def dtype(self) -> DTypeProtocol:
        """Return the array's dtype."""
        ...

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        ...

    @property
    def size(self) -> int:
        """Return total number of elements."""
        ...

    def tolist(self) -> list[float] | list[list[float]] | list[int] | list[list[int]]:
        """Convert array to nested Python lists.

        For 1D arrays: returns list[float] or list[int]
        For 2D arrays: returns list[list[float]] or list[list[int]]
        """
        ...

    def __len__(self) -> int:
        """Return length of first dimension."""
        ...

    def __getitem__(self, idx: int) -> float:
        """Index into the array."""
        ...


class NdArray1DProtocol(Protocol):
    """Protocol for 1D numpy arrays.

    More specific than NdArrayProtocol for cases where we know
    the array is 1-dimensional (e.g., retention times, m/z values).
    """

    @property
    def shape(self) -> tuple[int]:
        """Return (n,) shape for 1D array."""
        ...

    @property
    def dtype(self) -> DTypeProtocol:
        """Return the array's dtype."""
        ...

    def tolist(self) -> list[float]:
        """Convert 1D array to Python list of floats."""
        ...

    def __len__(self) -> int:
        """Return array length."""
        ...

    def __getitem__(self, idx: int) -> float:
        """Get element at index."""
        ...


class NdArray2DProtocol(Protocol):
    """Protocol for 2D numpy arrays.

    For matrices like intensity data (time x wavelength) or
    mass spectra (scan x m/z).
    """

    @property
    def shape(self) -> tuple[int, int]:
        """Return (rows, cols) shape for 2D array."""
        ...

    @property
    def dtype(self) -> DTypeProtocol:
        """Return the array's dtype."""
        ...

    def tolist(self) -> list[list[float]]:
        """Convert 2D array to nested Python lists."""
        ...

    def __len__(self) -> int:
        """Return number of rows."""
        ...

    def __getitem__(self, idx: int) -> NdArray1DProtocol:
        """Get row at index."""
        ...


__all__ = [
    "DTypeProtocol",
    "NdArray1DProtocol",
    "NdArray2DProtocol",
    "NdArrayProtocol",
]
