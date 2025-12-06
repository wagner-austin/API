"""Decoder functions for CSV chromatogram files.

Parses CSV/TSV chromatogram exports into typed structures.
Pure Python implementation with strict typing.
"""

from __future__ import annotations

from instrument_io._exceptions import CSVReadError, DecodingError
from instrument_io.types.chromatogram import ChromatogramData, ChromatogramStats


def _detect_delimiter(first_line: str) -> str:
    """Detect delimiter from first line of CSV file.

    Checks for tab first (TSV format), then comma (CSV format).

    Args:
        first_line: First line of the file (header row).

    Returns:
        Detected delimiter character.

    Raises:
        DecodingError: If no recognizable delimiter found.
    """
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    raise DecodingError("delimiter", "No tab or comma delimiter found in header")


def _parse_csv_line(line: str, delimiter: str) -> list[str]:
    """Parse a single CSV line into fields.

    Handles basic CSV parsing without quoted fields.
    Strips whitespace from each field.

    Args:
        line: Single line from CSV file.
        delimiter: Field delimiter character.

    Returns:
        List of field values as strings.
    """
    fields = line.split(delimiter)
    return [f.strip() for f in fields]


def _find_column_index(
    headers: list[str],
    column_name: str,
    source_path: str,
) -> int:
    """Find column index by name (case-insensitive).

    Args:
        headers: List of header names.
        column_name: Column name to find.
        source_path: Source file path for error messages.

    Returns:
        Zero-based column index.

    Raises:
        CSVReadError: If column not found.
    """
    column_lower = column_name.lower()
    for idx, header in enumerate(headers):
        if header.lower() == column_lower:
            return idx
    raise CSVReadError(
        source_path,
        f"Column '{column_name}' not found. Available columns: {headers}",
    )


def _parse_float_value(
    value: str,
    row_num: int,
    column_name: str,
    source_path: str,
) -> float:
    """Parse a string value to float.

    Args:
        value: String value to parse.
        row_num: Row number (1-based) for error messages.
        column_name: Column name for error messages.
        source_path: Source file path for error messages.

    Returns:
        Float value.

    Raises:
        CSVReadError: If value cannot be parsed as float.
    """
    stripped = value.strip()
    if not stripped:
        raise CSVReadError(
            source_path,
            f"Empty value in column '{column_name}' at row {row_num}",
        )

    # Handle common number formats
    # Remove thousands separators (comma when not delimiter)
    cleaned = stripped.replace(",", "")

    parsed: float
    # Check for integer format first
    if cleaned.lstrip("-").isdigit():
        parsed = float(int(cleaned))
    else:
        # Try float parsing
        float_candidate = cleaned
        # Handle scientific notation with lowercase 'e'
        if "e" in float_candidate.lower() and "e" not in float_candidate:
            float_candidate = float_candidate.replace("E", "e")

        # Validate float format before parsing
        parts = float_candidate.lstrip("-").split(".")
        if len(parts) > 2:
            raise CSVReadError(
                source_path,
                f"Invalid float '{stripped}' in column '{column_name}' at row {row_num}",
            )

        # Check for valid float characters
        valid_chars = set("0123456789.eE+-")
        if not all(c in valid_chars for c in float_candidate.replace("-", "", 1)):
            raise CSVReadError(
                source_path,
                f"Invalid float '{stripped}' in column '{column_name}' at row {row_num}",
            )

        parsed = float(float_candidate)

    return parsed


def _parse_float_column(
    rows: list[list[str]],
    column_idx: int,
    column_name: str,
    source_path: str,
) -> list[float]:
    """Parse all values in a column to floats.

    Args:
        rows: List of parsed row data (excluding header).
        column_idx: Column index to extract.
        column_name: Column name for error messages.
        source_path: Source file path for error messages.

    Returns:
        List of float values.

    Raises:
        CSVReadError: If any value cannot be parsed.
    """
    result: list[float] = []
    for row_num, row in enumerate(rows, start=2):  # Start at 2 (after header)
        if column_idx >= len(row):
            raise CSVReadError(
                source_path,
                f"Row {row_num} has {len(row)} columns, expected at least {column_idx + 1}",
            )
        value = row[column_idx]
        parsed = _parse_float_value(value, row_num, column_name, source_path)
        result.append(parsed)
    return result


def _compute_chromatogram_stats_from_data(
    retention_times: list[float],
    intensities: list[float],
    source_path: str,
) -> ChromatogramStats:
    """Compute statistics from chromatogram data.

    Args:
        retention_times: List of time points.
        intensities: List of intensity values.
        source_path: Source file path for error messages.

    Returns:
        ChromatogramStats TypedDict.

    Raises:
        CSVReadError: If data is empty or mismatched.
    """
    if not retention_times or not intensities:
        raise CSVReadError(source_path, "Empty chromatogram data")

    if len(retention_times) != len(intensities):
        raise CSVReadError(
            source_path,
            f"Mismatched lengths: {len(retention_times)} times vs {len(intensities)} intensities",
        )

    num_points = len(retention_times)
    rt_min = min(retention_times)
    rt_max = max(retention_times)
    rt_step_mean = (rt_max - rt_min) / (num_points - 1) if num_points > 1 else 0.0

    intensity_min = min(intensities)
    intensity_max = max(intensities)
    intensity_mean = sum(intensities) / num_points
    intensity_p99 = _percentile(intensities, 0.99)

    return ChromatogramStats(
        num_points=num_points,
        rt_min=rt_min,
        rt_max=rt_max,
        rt_step_mean=rt_step_mean,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        intensity_mean=intensity_mean,
        intensity_p99=intensity_p99,
    )


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile using linear interpolation.

    Args:
        values: List of numeric values.
        p: Percentile (0.0 to 1.0).

    Returns:
        Interpolated percentile value.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = p * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _make_chromatogram_data(
    retention_times: list[float],
    intensities: list[float],
) -> ChromatogramData:
    """Create ChromatogramData TypedDict.

    Args:
        retention_times: List of time points.
        intensities: List of intensity values.

    Returns:
        ChromatogramData TypedDict.
    """
    return ChromatogramData(
        retention_times=retention_times,
        intensities=intensities,
    )


__all__ = [
    "_compute_chromatogram_stats_from_data",
    "_detect_delimiter",
    "_find_column_index",
    "_make_chromatogram_data",
    "_parse_csv_line",
    "_parse_float_column",
    "_parse_float_value",
    "_percentile",
]
