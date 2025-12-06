"""Decoder functions for Word document (.docx) data.

Converts python-docx structures to typed data.
"""

from __future__ import annotations

from instrument_io._protocols.python_docx import (
    ParagraphProtocol,
    TableProtocol,
)
from instrument_io.types.common import CellValue


def _decode_cell_value(text: str) -> CellValue:
    """Decode a table cell value to CellValue type.

    Args:
        text: Cell text content.

    Returns:
        CellValue (str, int, float, bool, or None).
    """
    stripped = text.strip()
    if not stripped:
        return None

    # Try boolean
    lower = stripped.lower()
    if lower in ("true", "yes", "y"):
        return True
    if lower in ("false", "no", "n"):
        return False

    # Try integer
    if stripped.lstrip("-").isdigit():
        return int(stripped)

    # Try float (check for decimal point or scientific notation indicators)
    if _is_float_format(stripped):
        return float(stripped)

    # Return as string
    return stripped


def _is_float_format(value: str) -> bool:
    """Check if string is in float format.

    Args:
        value: String to check.

    Returns:
        True if string can be parsed as float.
    """
    if not value:
        return False

    # Simple validation: has decimal point, or scientific notation
    has_decimal = "." in value
    has_exp = "e" in value.lower()

    if not (has_decimal or has_exp):
        return False

    # Remove leading sign characters
    check_val = value.lstrip("-+")

    # For scientific notation
    if has_exp:
        parts = check_val.lower().split("e")
        if len(parts) != 2:
            return False
        # Check mantissa and exponent
        mantissa = parts[0].replace(".", "")
        exponent = parts[1].lstrip("-+")
        return mantissa.isdigit() and exponent.isdigit()

    # For decimal
    decimal_parts = check_val.split(".")
    if len(decimal_parts) != 2:
        return False

    left, right = decimal_parts
    # At least one side must have digits
    return (left.isdigit() or not left) and (right.isdigit() or not right) and bool(left or right)


def _decode_docx_table(table: TableProtocol) -> list[dict[str, CellValue]]:
    """Decode a Word table to list of row dictionaries.

    Assumes first row contains headers.

    Args:
        table: TableProtocol from python-docx.

    Returns:
        List of row dictionaries with typed cell values.
    """
    rows = table.rows

    # First row is headers
    header_cells = rows[0].cells
    headers: list[str] = [cell.text.strip() for cell in header_cells]

    # Decode data rows
    result: list[dict[str, CellValue]] = []
    for row in rows[1:]:
        cells = row.cells
        row_dict: dict[str, CellValue] = {}

        for i, header in enumerate(headers):
            if header and i < len(cells):
                cell_value = cells[i].text
                row_dict[header] = _decode_cell_value(cell_value)

        if row_dict:
            result.append(row_dict)

    return result


def _decode_paragraph_text(para: ParagraphProtocol) -> str:
    """Extract text from paragraph.

    Args:
        para: ParagraphProtocol from python-docx.

    Returns:
        Paragraph text content.
    """
    return para.text


def _get_heading_level(para: ParagraphProtocol) -> int | None:
    """Get heading level from paragraph style.

    Args:
        para: ParagraphProtocol from python-docx.

    Returns:
        Heading level (1-9) or None if not a heading.
    """
    style = para.style
    style_name: str = style.name
    if style_name.startswith("Heading "):
        parts = style_name.split(" ")
        if len(parts) >= 2 and parts[1].isdigit():
            level = int(parts[1])
            return level if 1 <= level <= 9 else None

    return None


__all__ = [
    "_decode_cell_value",
    "_decode_docx_table",
    "_decode_paragraph_text",
    "_get_heading_level",
]
