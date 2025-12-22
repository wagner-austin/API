"""ARFF file parsing.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

# Maximum rows to sample for analysis
MAX_SAMPLE_ROWS = 1000


def read_arff_header_and_sample(
    path: Path,
) -> tuple[tuple[str, ...], int, tuple[tuple[str, ...], ...]]:
    """Read ARFF header and sample rows.

    Args:
        path: Path to ARFF file.

    Returns:
        Tuple of (column names, total row count, sample rows).
    """
    columns: list[str] = []
    data_started = False
    data_rows: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped.lower().startswith("@attribute"):
                # Parse attribute name
                parts = stripped.split()
                if len(parts) >= 2:
                    attr_name = parts[1]
                    columns.append(attr_name)

            elif stripped.lower() == "@data":
                data_started = True

            elif data_started and stripped and not stripped.startswith("%"):
                data_rows.append(stripped)

    n_rows = len(data_rows)

    # Sample from both start and end to handle sorted datasets
    half_sample = MAX_SAMPLE_ROWS // 2
    if n_rows <= MAX_SAMPLE_ROWS:
        sample_lines = data_rows
    else:
        # Take half from start, half from end
        sample_lines = data_rows[:half_sample] + data_rows[-half_sample:]

    sample_rows: list[tuple[str, ...]] = []
    for line in sample_lines:
        row = tuple(line.split(","))
        sample_rows.append(row)

    return tuple(columns), n_rows, tuple(sample_rows)


__all__ = [
    "MAX_SAMPLE_ROWS",
    "read_arff_header_and_sample",
]
