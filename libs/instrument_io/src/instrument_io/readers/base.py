"""Protocol definitions for reader interfaces.

Defines the typed contracts that reader implementations must fulfill.
No recovery, no best-effort - failures propagate as exceptions.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Protocol

from instrument_io.types.chromatogram import DADData, EICData, TICData
from instrument_io.types.common import CellValue
from instrument_io.types.metadata import RunInfo
from instrument_io.types.spectrum import MSSpectrum


class ChromatogramReaderProtocol(Protocol):
    """Protocol for reading chromatogram data from instrument files.

    Implementations must provide methods for reading TIC, EIC, and DAD data.
    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if this reader supports the given path.

        Args:
            path: Path to check for format support.

        Returns:
            True if this reader can handle the format.
        """
        ...

    def read_tic(self, path: Path) -> TICData:
        """Read Total Ion Chromatogram from instrument file.

        Args:
            path: Path to instrument data file/directory.

        Returns:
            TICData TypedDict with meta, data, and stats.

        Raises:
            AgilentReadError: If reading fails.
            DecodingError: If data validation fails.
        """
        ...

    def read_eic(
        self,
        path: Path,
        target_mz: float,
        mz_tolerance: float,
    ) -> EICData:
        """Read Extracted Ion Chromatogram for target m/z.

        Args:
            path: Path to instrument data file/directory.
            target_mz: Target mass-to-charge ratio.
            mz_tolerance: Tolerance window in Daltons.

        Returns:
            EICData TypedDict with meta, params, data, and stats.

        Raises:
            AgilentReadError: If reading fails.
            DecodingError: If data validation fails.
        """
        ...

    def read_dad(self, path: Path) -> DADData:
        """Read full DAD (Diode Array Detector) data.

        Args:
            path: Path to instrument data file/directory.

        Returns:
            DADData TypedDict with wavelengths and intensity matrix.

        Raises:
            AgilentReadError: If reading fails.
            DecodingError: If data validation fails.
        """
        ...

    def find_runs(self, data_root: Path) -> list[RunInfo]:
        """Find all instrument run directories under a root path.

        Args:
            data_root: Root directory to search.

        Returns:
            List of RunInfo TypedDicts for each discovered run.

        Raises:
            AgilentReadError: If directory enumeration fails.
        """
        ...


class SpectrumReaderProtocol(Protocol):
    """Protocol for reading mass spectrum data.

    Implementations must provide methods for reading individual spectra
    and iterating over all spectra in a file.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if this reader supports the given path.

        Args:
            path: Path to check for format support.

        Returns:
            True if this reader can handle the format.
        """
        ...

    def read_spectrum(self, path: Path, scan_number: int) -> MSSpectrum:
        """Read a single spectrum by scan number.

        Args:
            path: Path to spectrum data file.
            scan_number: 1-based scan index.

        Returns:
            MSSpectrum TypedDict with meta, data, and stats.

        Raises:
            MzMLReadError: If reading fails.
            DecodingError: If data validation fails.
        """
        ...

    def iter_spectra(self, path: Path) -> Generator[MSSpectrum, None, None]:
        """Iterate over all spectra in file.

        Args:
            path: Path to spectrum data file.

        Yields:
            MSSpectrum TypedDict for each spectrum.

        Raises:
            MzMLReadError: If reading fails.
            DecodingError: If data validation fails.
        """
        ...

    def count_spectra(self, path: Path) -> int:
        """Count total number of spectra in file.

        Args:
            path: Path to spectrum data file.

        Returns:
            Total spectrum count.

        Raises:
            MzMLReadError: If reading fails.
        """
        ...


class ExcelReaderProtocol(Protocol):
    """Protocol for reading Excel files.

    Implementations provide sheet enumeration and typed data extraction.
    """

    def list_sheets(self, path: Path) -> list[str]:
        """List all sheet names in workbook.

        Args:
            path: Path to Excel file.

        Returns:
            List of sheet names.

        Raises:
            ExcelReadError: If reading fails.
        """
        ...

    def read_sheet(
        self,
        path: Path,
        sheet_name: str,
    ) -> list[dict[str, CellValue]]:
        """Read a single sheet as list of row dictionaries.

        Args:
            path: Path to Excel file.
            sheet_name: Name of sheet to read.

        Returns:
            List of row dictionaries with typed cell values.

        Raises:
            ExcelReadError: If reading fails.
        """
        ...

    def read_sheets(
        self,
        path: Path,
    ) -> dict[str, list[dict[str, CellValue]]]:
        """Read all sheets from workbook.

        Args:
            path: Path to Excel file.

        Returns:
            Dictionary mapping sheet names to row lists.

        Raises:
            ExcelReadError: If reading fails.
        """
        ...


__all__ = [
    "ChromatogramReaderProtocol",
    "ExcelReaderProtocol",
    "SpectrumReaderProtocol",
]
