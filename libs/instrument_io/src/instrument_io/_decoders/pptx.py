"""Decoder functions for PowerPoint presentation (.pptx) data.

Converts python-pptx structures to typed data.
"""

from __future__ import annotations

from instrument_io._protocols.python_pptx import (
    SlideProtocol,
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


def _decode_pptx_table(table: TableProtocol) -> list[dict[str, CellValue]]:
    """Decode a PowerPoint table to list of row dictionaries.

    Assumes first row contains headers.

    Args:
        table: TableProtocol from python-pptx.

    Returns:
        List of row dictionaries with typed cell values.
    """
    rows = table.rows

    # First row is headers
    header_cells = rows[0].cells
    headers: list[str] = [cell.text.strip() for cell in header_cells]

    # Decode data rows (skip first row which is headers)
    result: list[dict[str, CellValue]] = []
    for row_idx in range(1, len(rows)):
        row = rows[row_idx]
        cells = row.cells
        row_dict: dict[str, CellValue] = {}

        for i, header in enumerate(headers):
            if header and i < len(cells):
                cell_value = cells[i].text
                row_dict[header] = _decode_cell_value(cell_value)

        if row_dict:
            result.append(row_dict)

    return result


def _extract_slide_text(slide: SlideProtocol) -> str:
    """Extract all text from a slide.

    Args:
        slide: SlideProtocol from python-pptx.

    Returns:
        Slide text content (text frames concatenated).
    """
    text_parts: list[str] = []

    for shape in slide.shapes:
        if shape.has_text_frame:
            text = shape.text_frame.text
            if text:
                text_parts.append(text)

    return "\n".join(text_parts)


def _extract_slide_title(slide: SlideProtocol) -> str:
    """Extract title from slide.

    Args:
        slide: SlideProtocol from python-pptx.

    Returns:
        Slide title or empty string if no title.
    """
    for shape in slide.shapes:
        if shape.has_text_frame:
            # First text shape is typically the title
            text = shape.text_frame.text
            if text:
                return text.strip()

    return ""


__all__ = [
    "_decode_cell_value",
    "_decode_pptx_table",
    "_extract_slide_text",
    "_extract_slide_title",
]
