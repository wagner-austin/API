"""Protocol definitions for pyteomics library.

Provides type-safe interfaces to pyteomics mzML/mzXML readers
without importing pyteomics directly.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from types import TracebackType
from typing import Protocol

from instrument_io._protocols.numpy import NdArrayProtocol

# Type alias for pyteomics spectrum dict values.
# Values can be arrays, scalars, nested dicts, or lists of dicts.
# Following the refactor doc pattern for recursive types.
SpectrumValue = (
    NdArrayProtocol
    | float
    | int
    | str
    | bool
    | list["SpectrumValue"]
    | dict[str, "SpectrumValue"]
    | None
)


class SpectrumDictProtocol(Protocol):
    """Protocol for pyteomics spectrum dictionary.

    pyteomics returns dicts with known string keys containing
    spectrum data and metadata. Key arrays are 'm/z array' and
    'intensity array'. Nested structures include 'scanList' and
    'precursor' with dict/list values.
    """

    def __getitem__(self, key: str) -> SpectrumValue:
        """Get value by key."""
        ...

    def get(self, key: str) -> SpectrumValue:
        """Get value by key, returns None if not found."""
        ...

    def keys(self) -> Generator[str, None, None]:
        """Return generator over keys."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        ...


class MzMLIteratorProtocol(Protocol):
    """Protocol for iterating over mzML spectra."""

    def __iter__(self) -> Generator[SpectrumDictProtocol, None, None]:
        """Iterate over spectra."""
        ...

    def __next__(self) -> SpectrumDictProtocol:
        """Get next spectrum."""
        ...


class MzMLReaderProtocol(Protocol):
    """Protocol for pyteomics.mzml.MzML reader.

    Supports both iteration and context manager protocol.
    """

    def __iter__(self) -> Generator[SpectrumDictProtocol, None, None]:
        """Iterate over all spectra."""
        ...

    def __enter__(self) -> MzMLReaderProtocol:
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


class MzXMLReaderProtocol(Protocol):
    """Protocol for pyteomics.mzxml.MzXML reader.

    Similar interface to MzMLReaderProtocol for the older mzXML format.
    """

    def __iter__(self) -> Generator[SpectrumDictProtocol, None, None]:
        """Iterate over all scans."""
        ...

    def __enter__(self) -> MzXMLReaderProtocol:
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


def _open_mzml(path: Path) -> MzMLReaderProtocol:
    """Open mzML file via pyteomics with strict typing.

    Passes path string to pyteomics which handles file lifecycle internally.
    The returned reader implements context manager for proper cleanup.
    Uses use_index=False to avoid index-related warnings for streaming access.

    Args:
        path: Path to .mzML file.

    Returns:
        MzMLReaderProtocol for iterating over spectra.

    Raises:
        Exception: If pyteomics fails to open the file.
    """
    mod = __import__("pyteomics.mzml", fromlist=["MzML"])
    reader: MzMLReaderProtocol = mod.MzML(str(path), use_index=False)
    return reader


def _open_mzxml(path: Path) -> MzXMLReaderProtocol:
    """Open mzXML file via pyteomics with strict typing.

    Passes path string to pyteomics which handles file lifecycle internally.
    The returned reader implements context manager for proper cleanup.
    Uses use_index=False to avoid index-related warnings for streaming access.

    Args:
        path: Path to .mzXML file.

    Returns:
        MzXMLReaderProtocol for iterating over scans.

    Raises:
        Exception: If pyteomics fails to open the file.
    """
    mod = __import__("pyteomics.mzxml", fromlist=["MzXML"])
    reader: MzXMLReaderProtocol = mod.MzXML(str(path), use_index=False)
    return reader


__all__ = [
    "MzMLReaderProtocol",
    "MzXMLReaderProtocol",
    "SpectrumDictProtocol",
    "_open_mzml",
    "_open_mzxml",
]
