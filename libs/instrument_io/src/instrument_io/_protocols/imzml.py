"""Protocol definitions for pyimzML library.

Provides type-safe interfaces to pyimzML ImzMLParser
without importing pyimzML directly.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Protocol

from instrument_io._protocols.numpy import NdArray1DProtocol


class ImzMLParserProtocol(Protocol):
    """Protocol for pyimzML.ImzMLParser.

    Provides typed access to imzML imaging mass spectrometry data.
    Supports context manager and spectrum iteration.
    """

    @property
    def coordinates(self) -> list[tuple[int, int, int]]:
        """List of (x, y, z) coordinates for all pixels."""
        ...

    @property
    def polarity(self) -> str:
        """Ion polarity: 'positive', 'negative', or 'mixed'."""
        ...

    @property
    def spectrum_mode(self) -> str:
        """Spectrum mode: 'centroid' or 'profile'."""
        ...

    def getspectrum(self, index: int) -> tuple[NdArray1DProtocol, NdArray1DProtocol]:
        """Get m/z and intensity arrays for spectrum at index.

        Args:
            index: 0-based spectrum index.

        Returns:
            Tuple of (mz_array, intensity_array).
        """
        ...

    def __enter__(self) -> ImzMLParserProtocol:
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


def _open_imzml(path: Path) -> ImzMLParserProtocol:
    """Open imzML file via pyimzML with strict typing.

    Passes path string to pyimzML which handles file lifecycle internally.
    The returned parser implements context manager for proper cleanup.

    Args:
        path: Path to .imzML file.

    Returns:
        ImzMLParserProtocol for accessing spectra.

    Raises:
        FileNotFoundError: If file does not exist.
        Exception: If pyimzML fails to open the file.
    """
    mod = __import__("pyimzml.ImzMLParser", fromlist=["ImzMLParser"])
    parser: ImzMLParserProtocol = mod.ImzMLParser(str(path))
    return parser


__all__ = [
    "ImzMLParserProtocol",
    "_open_imzml",
]
