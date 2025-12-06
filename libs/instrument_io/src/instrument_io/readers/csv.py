"""CSV chromatogram file reader implementation.

Provides typed reading of CSV-exported chromatogram files with automatic
delimiter detection and flexible column mapping.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._decoders.csv import (
    _compute_chromatogram_stats_from_data,
    _detect_delimiter,
    _find_column_index,
    _make_chromatogram_data,
    _parse_csv_line,
    _parse_float_column,
)
from instrument_io._exceptions import CSVReadError
from instrument_io.types.chromatogram import (
    ChromatogramMeta,
    TICData,
)
from instrument_io.types.common import SignalType


def _is_csv_file(path: Path) -> bool:
    """Check if path is a CSV or TSV file."""
    return path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}


def _read_file_lines(path: Path) -> list[str]:
    """Read all non-empty lines from file.

    Args:
        path: Path to CSV file.

    Returns:
        List of non-empty, stripped lines.

    Raises:
        CSVReadError: If file cannot be read.
    """
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _build_chromatogram_meta(
    source_path: str,
    signal_type: SignalType,
    detector: str,
) -> ChromatogramMeta:
    """Build ChromatogramMeta with default values for unknown fields.

    Args:
        source_path: Path to source file.
        signal_type: Signal type literal.
        detector: Detector name string.

    Returns:
        ChromatogramMeta TypedDict.
    """
    return ChromatogramMeta(
        source_path=source_path,
        instrument="",
        method_name="",
        sample_name="",
        acquisition_date="",
        signal_type=signal_type,
        detector=detector,
    )


class CSVChromatogramReader:
    """Reader for CSV-exported chromatogram files.

    Reads chromatogram data from CSV/TSV files with configurable column
    names. Supports automatic delimiter detection and flexible column mapping.

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is a supported CSV/TSV file.

        Args:
            path: Path to check.

        Returns:
            True if path is a CSV, TSV, or TXT file.
        """
        return _is_csv_file(path)

    def detect_columns(self, path: Path, delimiter: str | None = None) -> list[str]:
        """Detect column names in CSV file.

        Args:
            path: Path to CSV file.
            delimiter: Column delimiter (auto-detected if None).

        Returns:
            List of column names from header row.

        Raises:
            CSVReadError: If file cannot be read or has no header.
        """
        source_path = str(path)

        lines = _read_file_lines(path)
        if not lines:
            raise CSVReadError(source_path, "Empty file")

        detected_delimiter = delimiter if delimiter else _detect_delimiter(lines[0])
        headers = _parse_csv_line(lines[0], detected_delimiter)

        # Filter out empty column names (whitespace-only columns)
        non_empty_headers = [h for h in headers if h]
        if not non_empty_headers:
            raise CSVReadError(source_path, "No columns found in header")

        return non_empty_headers

    def detect_delimiter(self, path: Path) -> str:
        """Detect the delimiter used in a CSV file.

        Args:
            path: Path to CSV file.

        Returns:
            Detected delimiter character.

        Raises:
            CSVReadError: If file cannot be read or delimiter cannot be detected.
        """
        source_path = str(path)

        lines = _read_file_lines(path)
        if not lines:
            raise CSVReadError(source_path, "Empty file")

        return _detect_delimiter(lines[0])

    def read_chromatogram(
        self,
        path: Path,
        time_column: str = "Time",
        intensity_column: str = "Intensity",
        delimiter: str | None = None,
        signal_type: SignalType = "TIC",
    ) -> TICData:
        """Read chromatogram from CSV file.

        Args:
            path: Path to CSV file.
            time_column: Name of retention time column (case-insensitive).
            intensity_column: Name of intensity column (case-insensitive).
            delimiter: Column delimiter (auto-detected if None).
            signal_type: Signal type to assign in metadata.

        Returns:
            TICData with chromatogram data.

        Raises:
            CSVReadError: If file cannot be parsed or columns missing.
        """
        source_path = str(path)

        lines = _read_file_lines(path)
        if not lines:
            raise CSVReadError(source_path, "Empty file")

        if len(lines) < 2:
            raise CSVReadError(source_path, "File has header but no data rows")

        # Detect or use provided delimiter
        detected_delimiter = delimiter if delimiter else _detect_delimiter(lines[0])

        # Parse header
        headers = _parse_csv_line(lines[0], detected_delimiter)

        # Find column indices
        time_idx = _find_column_index(headers, time_column, source_path)
        intensity_idx = _find_column_index(headers, intensity_column, source_path)

        # Parse data rows
        data_rows: list[list[str]] = []
        for line in lines[1:]:
            row = _parse_csv_line(line, detected_delimiter)
            data_rows.append(row)

        # Extract columns
        retention_times = _parse_float_column(data_rows, time_idx, time_column, source_path)
        intensities = _parse_float_column(data_rows, intensity_idx, intensity_column, source_path)

        # Build typed structures
        meta = _build_chromatogram_meta(source_path, signal_type, "CSV")
        data = _make_chromatogram_data(retention_times, intensities)
        stats = _compute_chromatogram_stats_from_data(retention_times, intensities, source_path)

        return TICData(meta=meta, data=data, stats=stats)


__all__ = [
    "CSVChromatogramReader",
]
