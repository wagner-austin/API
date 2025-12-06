"""Protocol definitions for pyteomics.mgf library.

Provides type-safe interfaces to pyteomics MGF reader
without importing pyteomics directly.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import TracebackType
from typing import Literal, Protocol, overload

from instrument_io._protocols.numpy import NdArray1DProtocol

# Type alias for MGF params dict
MGFParamsDict = dict[str, str | float | int | list[int] | tuple[float | None, ...] | None]


class MGFSpectrumProtocol(Protocol):
    """Protocol for pyteomics MGF spectrum dictionary.

    pyteomics MGF returns dicts with keys:
    - 'm/z array': numpy array of m/z values (1D)
    - 'intensity array': numpy array of intensities (1D)
    - 'params': dict of spectrum parameters (title, pepmass, charge, etc.)

    Uses overloads to provide precise return types for known keys.
    """

    @overload
    def __getitem__(self, key: Literal["m/z array"]) -> NdArray1DProtocol: ...
    @overload
    def __getitem__(self, key: Literal["intensity array"]) -> NdArray1DProtocol: ...
    @overload
    def __getitem__(self, key: Literal["params"]) -> MGFParamsDict: ...
    @overload
    def __getitem__(self, key: str) -> NdArray1DProtocol | MGFParamsDict: ...

    def __getitem__(self, key: str) -> NdArray1DProtocol | MGFParamsDict:
        """Get value by key."""
        ...

    def get(
        self,
        key: str,
        default: NdArray1DProtocol | MGFParamsDict | None = None,
    ) -> NdArray1DProtocol | MGFParamsDict | None:
        """Get value by key with default."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        ...


class MGFReaderProtocol(Protocol):
    """Protocol for pyteomics.mgf.MGF reader.

    Supports both iteration and context manager protocol.
    Iteration yields MGFSpectrumProtocol dicts.
    """

    def __iter__(self) -> Generator[MGFSpectrumProtocol, None, None]:
        """Iterate over all spectra."""
        ...

    def __enter__(self) -> MGFReaderProtocol:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        ...


def _open_mgf(path: Path) -> MGFReaderProtocol:
    """Open MGF file via pyteomics with strict typing.

    Passes path string to pyteomics which handles file lifecycle internally.
    The returned reader implements context manager for proper cleanup.

    Args:
        path: Path to .mgf file.

    Returns:
        MGFReaderProtocol for iterating over spectra.

    Raises:
        FileNotFoundError: If file does not exist.
        Exception: If pyteomics fails to open the file.
    """
    mod = __import__("pyteomics.mgf", fromlist=["MGF"])
    reader: MGFReaderProtocol = mod.MGF(str(path))
    return reader


__all__ = [
    "MGFParamsDict",
    "MGFReaderProtocol",
    "MGFSpectrumProtocol",
    "_open_mgf",
]
