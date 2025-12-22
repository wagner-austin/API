"""CSV and data file parsing.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

# Maximum rows to sample for analysis
MAX_SAMPLE_ROWS = 1000


def detect_csv_delimiter(
    first_line: str,
    prefer_space: bool = False,
) -> Literal[",", ";", "\t", " "]:
    """Detect CSV delimiter from first line.

    Counts occurrences of each delimiter and returns the most frequent.
    Space is only selected if prefer_space is True (for .data files).

    Args:
        first_line: First line (header) of CSV file.
        prefer_space: If True, allow space as delimiter (for .data files).

    Returns:
        Detected delimiter character.
    """
    comma_count = first_line.count(",")
    semicolon_count = first_line.count(";")
    tab_count = first_line.count("\t")

    # For .data files, check if space is the likely delimiter
    if prefer_space:
        space_count = first_line.count(" ")
        # Only use space if no standard delimiters present
        no_standard_delimiters = comma_count == 0 and semicolon_count == 0 and tab_count == 0
        if no_standard_delimiters and space_count > 0:
            return " "

    # Return delimiter with highest count, preferring comma on ties
    if semicolon_count > comma_count and semicolon_count > tab_count:
        return ";"
    if tab_count > comma_count and tab_count > semicolon_count:
        return "\t"
    return ","


def strip_quotes(value: str) -> str:
    """Strip surrounding quotes from a value.

    Handles double quotes and single quotes.

    Args:
        value: String value that may have surrounding quotes.

    Returns:
        String with surrounding quotes removed.
    """
    stripped = value.strip()
    has_double_quotes = stripped.startswith('"') and stripped.endswith('"')
    has_single_quotes = stripped.startswith("'") and stripped.endswith("'")
    if len(stripped) >= 2 and (has_double_quotes or has_single_quotes):
        return stripped[1:-1]
    return stripped


def read_csv_header_and_sample(
    path: Path,
    encoding: Literal["utf-8", "utf-8-sig", "latin-1", "cp1252"],
) -> tuple[tuple[str, ...], int, tuple[tuple[str, ...], ...]]:
    """Read CSV header and sample rows.

    Uses Python's csv module to properly handle quoted fields.
    Automatically detects delimiter (comma, semicolon, or tab).

    Args:
        path: Path to CSV file.
        encoding: File encoding.

    Returns:
        Tuple of (column names, total row count, sample rows).
    """
    with open(path, encoding=encoding, newline="") as f:
        # Read first line to detect delimiter
        first_line = f.readline()
        if not first_line:
            return (), 0, ()

        delimiter = detect_csv_delimiter(first_line.strip())

        # Reset to start and use csv reader
        f.seek(0)
        reader = csv.reader(f, delimiter=delimiter)

        # Read header (guaranteed to exist since first_line was not empty)
        header_row = next(reader)
        columns = tuple(col.strip() for col in header_row)

        # Read sample rows and count total
        sample_rows: list[tuple[str, ...]] = []
        n_rows = 0
        for row in reader:
            n_rows += 1
            if len(sample_rows) < MAX_SAMPLE_ROWS:
                sample_rows.append(tuple(val.strip() for val in row))

    return columns, n_rows, tuple(sample_rows)


def read_data_header_and_sample(
    path: Path,
    encoding: Literal["utf-8", "utf-8-sig", "latin-1", "cp1252"],
) -> tuple[tuple[str, ...], int, tuple[tuple[str, ...], ...]]:
    """Read space-delimited .data file (no header row).

    These files typically have no header. Column names are auto-generated
    as X1, X2, ..., with the last column named 'class' for target detection.

    Streams the file to avoid loading entire file into memory.

    Args:
        path: Path to .data file.
        encoding: File encoding.

    Returns:
        Tuple of (column names, total row count, sample rows).
    """
    with open(path, encoding=encoding) as f:
        # Read first line to detect delimiter and count columns
        first_line = f.readline()
        if not first_line:
            return (), 0, ()

        first_line_stripped = first_line.strip()
        delimiter = detect_csv_delimiter(first_line_stripped, prefer_space=True)

        # Count columns from first row (split always returns at least one element)
        first_row_values = first_line_stripped.split(delimiter)
        n_cols = len(first_row_values)

        # Generate column names: X1, X2, ..., Xn-1, class
        # Last column is typically the class/target
        column_names: list[str] = []
        for i in range(n_cols - 1):
            column_names.append(f"X{i + 1}")
        column_names.append("class")
        columns = tuple(column_names)

        # Process first row as data (no header in .data files)
        sample_rows: list[tuple[str, ...]] = []
        first_row = tuple(strip_quotes(val) for val in first_row_values)
        sample_rows.append(first_row)
        n_rows = 1

        # Stream remaining lines, counting all but only keeping samples
        for line in f:
            n_rows += 1
            if len(sample_rows) < MAX_SAMPLE_ROWS:
                raw_row = line.strip().split(delimiter)
                row = tuple(strip_quotes(val) for val in raw_row)
                sample_rows.append(row)

    return columns, n_rows, tuple(sample_rows)


__all__ = [
    "MAX_SAMPLE_ROWS",
    "detect_csv_delimiter",
    "read_csv_header_and_sample",
    "read_data_header_and_sample",
    "strip_quotes",
]
