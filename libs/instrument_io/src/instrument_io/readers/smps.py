"""SMPS .rps file reader implementation.

Provides typed reading of SMPS (Scanning Mobility Particle Sizer) data files.
All methods raise exceptions on failure - no recovery or fallbacks.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._decoders.smps import (
    _decode_smps_data,
    _decode_smps_full,
    _decode_smps_metadata,
)
from instrument_io._exceptions import SMPSReadError
from instrument_io.types.common import CellValue
from instrument_io.types.smps import SMPSData, SMPSMetadata


def _is_rps_file(path: Path) -> bool:
    """Check if path is an SMPS .rps file."""
    return path.is_file() and path.suffix.lower() == ".rps"


def _read_lines(path: Path) -> list[str]:
    """Read all lines from file.

    Args:
        path: Path to .rps file.

    Returns:
        List of text lines (newlines stripped).

    Raises:
        SMPSReadError: If reading fails.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return [line.rstrip("\r\n") for line in f]
    except UnicodeDecodeError:
        # Try cp1252 encoding (common for Windows-generated files)
        try:
            with path.open("r", encoding="cp1252") as f:
                return [line.rstrip("\r\n") for line in f]
        except (UnicodeDecodeError, OSError) as e:
            raise SMPSReadError(str(path), f"Failed to read file: {e}") from e
    except OSError as e:
        raise SMPSReadError(str(path), f"Failed to read file: {e}") from e


class SMPSReader:
    """Reader for SMPS .rps files.

    SMPS files are tab-delimited text files containing particle size distribution data.
    File structure:
    - Line 0: timestamp, time, instrument name
    - Lines 1-2: parameter names and values
    - Line 3: data column headers
    - Lines 4+: data rows

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is an SMPS .rps file.

        Args:
            path: Path to check.

        Returns:
            True if path is an SMPS .rps file.
        """
        return _is_rps_file(path)

    def read_metadata(self, path: Path) -> SMPSMetadata:
        """Read SMPS metadata from file header.

        Args:
            path: Path to .rps file.

        Returns:
            SMPSMetadata TypedDict with measurement parameters.

        Raises:
            SMPSReadError: If reading fails.
        """
        if not path.exists():
            raise SMPSReadError(str(path), "File does not exist")

        if not _is_rps_file(path):
            raise SMPSReadError(str(path), "Not an SMPS .rps file")

        lines = _read_lines(path)
        return _decode_smps_metadata(lines)

    def read_data(self, path: Path) -> list[dict[str, CellValue]]:
        """Read SMPS data rows (without metadata).

        Args:
            path: Path to .rps file.

        Returns:
            List of row dictionaries with typed cell values.

        Raises:
            SMPSReadError: If reading fails.
        """
        if not path.exists():
            raise SMPSReadError(str(path), "File does not exist")

        if not _is_rps_file(path):
            raise SMPSReadError(str(path), "Not an SMPS .rps file")

        lines = _read_lines(path)
        return _decode_smps_data(lines)

    def read_full(self, path: Path) -> SMPSData:
        """Read complete SMPS file (metadata + data).

        Args:
            path: Path to .rps file.

        Returns:
            SMPSData TypedDict with metadata and data rows.

        Raises:
            SMPSReadError: If reading fails.
        """
        if not path.exists():
            raise SMPSReadError(str(path), "File does not exist")

        if not _is_rps_file(path):
            raise SMPSReadError(str(path), "Not an SMPS .rps file")

        lines = _read_lines(path)
        return _decode_smps_full(lines)


__all__ = [
    "SMPSReader",
]
