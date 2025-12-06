"""Decoder functions for PDF data via pdfplumber.

Converts PDF table data to typed structures.
"""

from __future__ import annotations

from instrument_io.types.common import CellValue


def _decode_pdf_cell(value: str | None) -> CellValue:
    """Decode a PDF table cell value to CellValue type.

    Args:
        value: Raw cell value from pdfplumber (string or None).

    Returns:
        CellValue (str, int, float, bool, or None).
    """
    if value is None:
        return None

    # Empty strings become None
    stripped = value.strip()
    if not stripped:
        return None

    # Try to parse as boolean
    lower = stripped.lower()
    if lower in ("true", "yes", "y"):
        return True
    if lower in ("false", "no", "n"):
        return False

    # Parse as number if it looks numeric
    if _is_integer_string(stripped):
        return int(stripped)

    if _is_float_string(stripped):
        return float(stripped)

    # Return as string
    return stripped


def _is_integer_string(value: str) -> bool:
    """Check if string represents a valid integer.

    Args:
        value: String to check.

    Returns:
        True if string is a valid integer.
    """
    if not value:
        return False

    # Handle negative numbers
    if value[0] == "-":
        if len(value) == 1:
            return False
        return value[1:].isdigit()

    return value.isdigit()


def _is_float_string(value: str) -> bool:
    """Check if string represents a valid float.

    Args:
        value: String to check.

    Returns:
        True if string is a valid float.
    """
    if not value or value == "." or value == "-":
        return False

    # Handle negative numbers
    start_idx = 1 if value[0] == "-" else 0

    # Must have at least one digit
    has_digit = False
    has_decimal = False

    for i in range(start_idx, len(value)):
        char = value[i]
        if char.isdigit():
            has_digit = True
        elif char == ".":
            if has_decimal:  # Multiple decimal points
                return False
            has_decimal = True
        else:
            return False

    return has_digit and has_decimal


def _decode_pdf_row(row: list[str | None]) -> list[CellValue]:
    """Decode a PDF table row to typed cell values.

    Args:
        row: List of cell values from pdfplumber table row.

    Returns:
        List of typed cell values.
    """
    return [_decode_pdf_cell(cell) for cell in row]


def _decode_pdf_table(
    table: list[list[str | None]],
) -> list[dict[str, CellValue]]:
    """Decode a PDF table to list of row dictionaries.

    Assumes first row contains headers.

    Args:
        table: Table from pdfplumber (list of rows, each row is list of cells).

    Returns:
        List of row dictionaries with typed cell values.
    """
    if not table:
        return []

    # First row is headers
    header_row = table[0]
    headers: list[str] = []
    for cell in header_row:
        if cell is None:
            headers.append("")
        else:
            headers.append(cell.strip())

    # Decode data rows
    result: list[dict[str, CellValue]] = []
    for row in table[1:]:
        decoded_row = _decode_pdf_row(row)
        row_dict: dict[str, CellValue] = {}
        for i, header in enumerate(headers):
            if header and i < len(decoded_row):
                row_dict[header] = decoded_row[i]
        if row_dict:  # Skip empty rows
            result.append(row_dict)

    return result


__all__ = [
    "_decode_pdf_cell",
    "_decode_pdf_row",
    "_decode_pdf_table",
    "_is_float_string",
    "_is_integer_string",
]
