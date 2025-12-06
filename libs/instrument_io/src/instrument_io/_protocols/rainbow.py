"""Protocol definitions for rainbow-api library.

Provides type-safe interfaces to rainbow's DataFile and DataDirectory
classes without importing rainbow directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from instrument_io._protocols.numpy import NdArrayProtocol


class DataFileProtocol(Protocol):
    """Protocol for rainbow.DataFile.

    Represents a single data file within an Agilent .D directory.
    Contains chromatogram or spectrum data with associated metadata.

    Key attributes:
        xlabels: Retention times (1D array, minutes)
        ylabels: Wavelengths or m/z values (1D array, may be empty for TIC)
        data: Intensity matrix (2D for DAD/MS, or 1D-like for single channel)
        detector: Detector name string (e.g., "TIC", "DAD1A", "MS")
        name: Filename within the .D directory
    """

    @property
    def xlabels(self) -> NdArrayProtocol:
        """Return retention time array (x-axis)."""
        ...

    @property
    def ylabels(self) -> NdArrayProtocol:
        """Return wavelength/m/z array (y-axis for 2D data)."""
        ...

    @property
    def data(self) -> NdArrayProtocol:
        """Return intensity data array."""
        ...

    @property
    def detector(self) -> str:
        """Return detector name."""
        ...

    @property
    def name(self) -> str:
        """Return filename."""
        ...

    def get_info(self) -> str:
        """Return human-readable info string."""
        ...


class DataDirectoryProtocol(Protocol):
    """Protocol for rainbow.DataDirectory.

    Represents an Agilent .D folder containing multiple data files.
    Provides access to individual files and directory-level metadata.
    """

    @property
    def datafiles(self) -> list[DataFileProtocol]:
        """Return list of data files in the directory."""
        ...

    @property
    def directory(self) -> str:
        """Return the directory path as string."""
        ...

    def get_file(self, name: str) -> DataFileProtocol | None:
        """Get data file by name, or None if not found."""
        ...

    def get_detector(self, detector: str) -> list[DataFileProtocol]:
        """Get all files for a specific detector type."""
        ...


class _RawDataDirectoryProtocol(Protocol):
    """Protocol for raw rainbow.DataDirectory (actual rainbow interface)."""

    @property
    def name(self) -> str:
        """Return directory basename."""
        ...

    @property
    def datafiles(self) -> list[DataFileProtocol]:
        """Return list of data files."""
        ...

    @property
    def by_detector(self) -> dict[str, list[DataFileProtocol]]:
        """Return detector -> files mapping."""
        ...

    @property
    def by_name(self) -> dict[str, DataFileProtocol]:
        """Return name -> file mapping."""
        ...


class _RainbowReadFn(Protocol):
    """Protocol for rainbow.read function."""

    def __call__(self, path: str) -> _RawDataDirectoryProtocol:
        """Read an Agilent .D directory."""
        ...


class _DataDirectoryAdapter:
    """Adapter wrapping rainbow DataDirectory to match DataDirectoryProtocol."""

    def __init__(self, raw: _RawDataDirectoryProtocol, path: Path) -> None:
        self._raw = raw
        self._path = path

    @property
    def datafiles(self) -> list[DataFileProtocol]:
        """Return list of data files in the directory."""
        return self._raw.datafiles

    @property
    def directory(self) -> str:
        """Return the directory path as string."""
        return str(self._path)

    def get_file(self, name: str) -> DataFileProtocol | None:
        """Get data file by name, or None if not found."""
        return self._raw.by_name.get(name.upper())

    def get_detector(self, detector: str) -> list[DataFileProtocol]:
        """Get all files for a specific detector type."""
        return self._raw.by_detector.get(detector.upper(), [])


def _load_data_directory(path: Path) -> DataDirectoryProtocol:
    """Load Agilent .D directory via rainbow with strict typing.

    Args:
        path: Path to .D directory.

    Returns:
        DataDirectoryProtocol representing the loaded data.

    Raises:
        Exception: If rainbow fails to read the directory.
    """
    rainbow_mod = __import__("rainbow")
    read_fn: _RainbowReadFn = rainbow_mod.read
    raw = read_fn(str(path))
    adapter: DataDirectoryProtocol = _DataDirectoryAdapter(raw, path)
    return adapter


__all__ = [
    "DataDirectoryProtocol",
    "DataFileProtocol",
    "_load_data_directory",
]
