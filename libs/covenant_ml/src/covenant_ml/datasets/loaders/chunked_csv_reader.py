"""Chunked CSV reader with progress reporting using Polars.

Reads large CSV files with progress reporting using Polars' streaming
capabilities. Converts to list-of-lists format for compatibility with
existing loader infrastructure.

Internal module - used by csv_loader and timeseries_csv_loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import FileEncoding, LoadPhase, LoadProgress

# Default batch size for reading CSV files (100,000 rows per batch)
DEFAULT_BATCH_SIZE: int = 100_000

# Minimum file size to trigger progress reporting (1 MB)
PROGRESS_THRESHOLD_BYTES: int = 1_024 * 1_024


class _PolarsDataFrameProtocol(Protocol):
    """Protocol for Polars DataFrame with required operations.

    Defines minimal interface for type-safe DataFrame operations.
    """

    @property
    def columns(self) -> list[str]:
        """Return column names."""
        ...

    @property
    def height(self) -> int:
        """Return number of rows."""
        ...

    @property
    def width(self) -> int:
        """Return number of columns."""
        ...

    def row(self, index: int) -> tuple[str, ...]:
        """Return single row as tuple."""
        ...

    def iter_rows(self) -> list[tuple[str, ...]]:
        """Iterate over rows as tuples."""
        ...


class _PolarsScanCSVProtocol(Protocol):
    """Protocol for Polars scan_csv lazy reader."""

    def collect(self) -> _PolarsDataFrameProtocol:
        """Materialize lazy frame into DataFrame."""
        ...


class _PolarsReadCSVProtocol(Protocol):
    """Protocol for Polars read_csv function."""

    def __call__(
        self,
        source: str | Path,
        encoding: str,
        infer_schema_length: int,
        n_rows: int | None = None,
    ) -> _PolarsDataFrameProtocol:
        """Read CSV file into DataFrame."""
        ...


def _get_polars_read_csv() -> _PolarsReadCSVProtocol:
    """Get Polars read_csv function with typing.

    Returns:
        Typed read_csv function.
    """
    polars_mod = __import__("polars")
    read_fn: _PolarsReadCSVProtocol = polars_mod.read_csv
    return read_fn


def _make_progress(
    phase: LoadPhase,
    bytes_read: int,
    bytes_total: int,
    rows_processed: int,
    rows_total: int,
    message: str,
) -> LoadProgress:
    """Create a LoadProgress dict with computed percent_complete.

    Args:
        phase: Current loading phase.
        bytes_read: Number of bytes read from source file.
        bytes_total: Total bytes in source file.
        rows_processed: Number of rows processed so far.
        rows_total: Total rows (0 if unknown).
        message: Human-readable status message.

    Returns:
        LoadProgress with computed percent_complete.
    """
    if bytes_total > 0:
        percent = (bytes_read / bytes_total) * 100.0
    elif rows_total > 0:
        percent = (rows_processed / rows_total) * 100.0
    else:
        percent = 0.0

    return LoadProgress(
        phase=phase,
        bytes_read=bytes_read,
        bytes_total=bytes_total,
        rows_processed=rows_processed,
        rows_total=rows_total,
        percent_complete=min(percent, 100.0),
        message=message,
    )


def _report_progress(
    callback: ProgressCallbackProtocol | None,
    phase: LoadPhase,
    bytes_read: int,
    bytes_total: int,
    rows_processed: int,
    rows_total: int,
    message: str,
) -> None:
    """Report progress if callback is provided.

    Args:
        callback: Optional progress callback.
        phase: Current loading phase.
        bytes_read: Number of bytes read from source file.
        bytes_total: Total bytes in source file.
        rows_processed: Number of rows processed so far.
        rows_total: Total rows (0 if unknown).
        message: Human-readable status message.
    """
    if callback is not None:
        progress = _make_progress(
            phase=phase,
            bytes_read=bytes_read,
            bytes_total=bytes_total,
            rows_processed=rows_processed,
            rows_total=rows_total,
            message=message,
        )
        callback(progress)


def _convert_encoding(encoding: FileEncoding) -> str:
    """Convert FileEncoding literal to Polars encoding string.

    Args:
        encoding: FileEncoding literal value.

    Returns:
        Encoding string compatible with Polars.
    """
    # Polars uses standard encoding names
    encoding_map: dict[str, str] = {
        "utf-8": "utf8",
        "utf-8-sig": "utf8",  # Polars handles BOM automatically
        "latin-1": "utf8-lossy",  # Polars doesn't support latin-1 directly
        "cp1252": "utf8-lossy",  # Use lossy fallback
    }
    return encoding_map.get(encoding, "utf8")


def read_csv_with_progress(
    file_path: Path,
    encoding: FileEncoding,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Read CSV file with progress reporting.

    Uses Polars for efficient reading and converts to list format
    for compatibility with existing loader infrastructure.

    Args:
        file_path: Path to CSV file.
        encoding: File encoding to use.
        progress_callback: Optional callback for progress updates.

    Returns:
        Tuple of (headers, rows) where headers is list of column names
        and rows is list of row values as strings.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If no data rows found.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    file_size = file_path.stat().st_size
    polars_encoding = _convert_encoding(encoding)

    # Report initial progress for large files
    if file_size >= PROGRESS_THRESHOLD_BYTES:
        _report_progress(
            callback=progress_callback,
            phase="reading",
            bytes_read=0,
            bytes_total=file_size,
            rows_processed=0,
            rows_total=0,
            message=f"Reading {file_path.name} ({file_size / (1024 * 1024 * 1024):.2f} GB)...",
        )

    # Read CSV using Polars - all columns as strings for compatibility
    read_csv = _get_polars_read_csv()
    df: _PolarsDataFrameProtocol = read_csv(
        file_path,
        encoding=polars_encoding,
        infer_schema_length=0,  # Treat all as strings
    )

    n_rows = df.height
    if n_rows == 0:
        raise ValueError(f"No data rows found in {file_path}")

    # Report progress after reading
    if file_size >= PROGRESS_THRESHOLD_BYTES:
        _report_progress(
            callback=progress_callback,
            phase="reading",
            bytes_read=file_size,
            bytes_total=file_size,
            rows_processed=n_rows,
            rows_total=n_rows,
            message=f"Read {n_rows:,} rows, converting to lists...",
        )

    # Extract headers (strip whitespace for clean column names)
    headers: list[str] = [h.strip() for h in df.columns]

    # Convert to list of lists
    # Use row-based progress (bytes=0) since file is already read
    _report_progress(
        callback=progress_callback,
        phase="parsing",
        bytes_read=0,
        bytes_total=0,
        rows_processed=0,
        rows_total=n_rows,
        message="Converting to row format...",
    )

    rows: list[list[str]] = []
    row_tuples = df.iter_rows()
    for i, row_tuple in enumerate(row_tuples):
        row_list: list[str] = [str(val) if val is not None else "" for val in row_tuple]
        rows.append(row_list)

        # Report progress periodically for large files
        # Use row-based progress (bytes=0) since file is already read
        if progress_callback is not None and (i + 1) % DEFAULT_BATCH_SIZE == 0:
            _report_progress(
                callback=progress_callback,
                phase="parsing",
                bytes_read=0,
                bytes_total=0,
                rows_processed=i + 1,
                rows_total=n_rows,
                message=f"Converted {i + 1:,} / {n_rows:,} rows...",
            )

    # Final progress - use row-based for consistency
    _report_progress(
        callback=progress_callback,
        phase="parsing",
        bytes_read=0,
        bytes_total=0,
        rows_processed=n_rows,
        rows_total=n_rows,
        message=f"Loaded {n_rows:,} rows from {file_path.name}",
    )

    return headers, rows


def read_csv_to_dataframe(
    file_path: Path,
    encoding: FileEncoding,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> _PolarsDataFrameProtocol:
    """Read CSV file to Polars DataFrame with progress reporting.

    Returns the raw DataFrame for more efficient processing pipelines.

    Args:
        file_path: Path to CSV file.
        encoding: File encoding to use.
        progress_callback: Optional callback for progress updates.

    Returns:
        Polars DataFrame with all columns as strings.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If no data rows found.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    file_size = file_path.stat().st_size
    polars_encoding = _convert_encoding(encoding)

    # Report initial progress
    _report_progress(
        callback=progress_callback,
        phase="reading",
        bytes_read=0,
        bytes_total=file_size,
        rows_processed=0,
        rows_total=0,
        message=f"Reading {file_path.name}...",
    )

    read_csv = _get_polars_read_csv()
    df: _PolarsDataFrameProtocol = read_csv(
        file_path,
        encoding=polars_encoding,
        infer_schema_length=0,
    )

    if df.height == 0:
        raise ValueError(f"No data rows found in {file_path}")

    _report_progress(
        callback=progress_callback,
        phase="reading",
        bytes_read=file_size,
        bytes_total=file_size,
        rows_processed=df.height,
        rows_total=df.height,
        message=f"Read {df.height:,} rows from {file_path.name}",
    )

    return df


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "PROGRESS_THRESHOLD_BYTES",
    "_make_progress",
    "read_csv_to_dataframe",
    "read_csv_with_progress",
]
