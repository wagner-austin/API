"""Tests for _protocols.openpyxl module."""

from __future__ import annotations

from pathlib import Path

from instrument_io._protocols.openpyxl import (
    _create_alignment,
    _create_font,
    _create_table,
    _create_workbook,
    _get_column_letter,
    _load_workbook,
)


def test_create_workbook() -> None:
    wb = _create_workbook()
    # Verify active sheet exists by accessing title
    _ = wb.active.title
    wb.close()


def test_load_workbook(tmp_path: Path) -> None:
    # Create a workbook first
    wb = _create_workbook()
    file_path = tmp_path / "test.xlsx"
    wb.save(file_path)
    wb.close()

    # Load it
    loaded = _load_workbook(file_path, read_only=False, data_only=False)
    # Verify active sheet exists by accessing title
    _ = loaded.active.title
    loaded.close()


def test_load_workbook_read_only(tmp_path: Path) -> None:
    wb = _create_workbook()
    file_path = tmp_path / "test.xlsx"
    wb.save(file_path)
    wb.close()

    loaded = _load_workbook(file_path, read_only=True, data_only=True)
    # Verify workbook loaded by checking first sheet name
    assert loaded.sheetnames[0] == "Sheet"
    loaded.close()


def test_get_column_letter() -> None:
    assert _get_column_letter(1) == "A"
    assert _get_column_letter(2) == "B"
    assert _get_column_letter(26) == "Z"
    assert _get_column_letter(27) == "AA"


def test_create_font() -> None:
    font = _create_font(bold=True, size=12.0)
    assert font.bold is True
    assert font.size == 12.0


def test_create_font_defaults() -> None:
    font = _create_font()
    assert font.bold is False
    assert font.size == 11.0


def test_create_alignment() -> None:
    alignment = _create_alignment(horizontal="center", vertical="top")
    assert alignment.horizontal == "center"
    assert alignment.vertical == "top"


def test_create_alignment_defaults() -> None:
    alignment = _create_alignment()
    assert alignment.horizontal == "general"
    assert alignment.vertical == "bottom"


def test_create_table() -> None:
    table = _create_table("MyTable", "A1:D10")
    assert table.name == "MyTable"
    assert table.ref == "A1:D10"


def test_workbook_create_sheet() -> None:
    wb = _create_workbook()
    ws = wb.create_sheet(title="NewSheet")
    assert ws.title == "NewSheet"
    assert "NewSheet" in wb.sheetnames
    wb.close()


def test_worksheet_cell_access() -> None:
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Test")
    assert ws.cell(row=1, column=1).value == "Test"
    wb.close()


def test_worksheet_column_dimensions() -> None:
    wb = _create_workbook()
    ws = wb.active
    col_letter = _get_column_letter(1)
    ws.column_dimensions[col_letter].width = 20.0
    assert ws.column_dimensions[col_letter].width == 20.0
    wb.close()


def test_workbook_save_and_close(tmp_path: Path) -> None:
    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Data")

    file_path = tmp_path / "test_save.xlsx"
    wb.save(file_path)
    wb.close()

    assert file_path.exists()


def test_create_pattern_fill() -> None:
    from instrument_io._protocols.openpyxl import _create_pattern_fill

    fill = _create_pattern_fill(start_color="FF0000")
    # openpyxl prepends alpha channel (00) to RGB colors
    assert fill.start_color.rgb == "00FF0000"
    assert fill.fill_type == "solid"


def test_create_pattern_fill_with_end_color() -> None:
    from instrument_io._protocols.openpyxl import _create_pattern_fill

    fill = _create_pattern_fill(start_color="FF0000", end_color="00FF00")
    # openpyxl prepends alpha channel (00) to RGB colors
    assert fill.start_color.rgb == "00FF0000"
    assert fill.end_color.rgb == "0000FF00"


def test_create_pattern_fill_custom_type() -> None:
    from instrument_io._protocols.openpyxl import _create_pattern_fill

    fill = _create_pattern_fill(start_color="FF0000", fill_type="solid")
    assert fill.fill_type == "solid"


def test_extract_header_strings() -> None:
    from instrument_io._protocols.openpyxl import _extract_header_strings

    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Header1")
    ws.cell(row=1, column=2, value="Header2")
    ws.cell(row=1, column=3, value=None)
    ws.cell(row=1, column=4, value=123)

    row = tuple(ws[1])
    headers = _extract_header_strings(row)

    assert headers == ["Header1", "Header2", "", "123"]
    wb.close()


def test_auto_adjust_column_widths() -> None:
    from instrument_io._protocols.openpyxl import _auto_adjust_column_widths

    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Short")
    ws.cell(row=2, column=1, value="This is a longer value")
    ws.cell(row=1, column=2, value="X")

    _auto_adjust_column_widths(ws, max_width=60, padding=2)

    # Column A should be wider than column B
    col_a_width = ws.column_dimensions["A"].width
    col_b_width = ws.column_dimensions["B"].width
    assert col_a_width > col_b_width
    wb.close()


def test_auto_adjust_column_widths_respects_max() -> None:
    from instrument_io._protocols.openpyxl import _auto_adjust_column_widths

    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="X" * 100)  # Very long value

    _auto_adjust_column_widths(ws, max_width=30, padding=2)

    col_a_width = ws.column_dimensions["A"].width
    assert col_a_width <= 30
    wb.close()


def test_auto_adjust_column_widths_with_none_values() -> None:
    from instrument_io._protocols.openpyxl import _auto_adjust_column_widths

    wb = _create_workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="Header")
    ws.cell(row=2, column=1, value=None)
    ws.cell(row=3, column=1, value="Data")

    _auto_adjust_column_widths(ws, max_width=60, padding=2)

    col_a_width = ws.column_dimensions["A"].width
    assert col_a_width > 0
    wb.close()


def test_auto_adjust_column_widths_empty_worksheet() -> None:
    """Test auto adjust on empty worksheet."""
    from instrument_io._protocols.openpyxl import _auto_adjust_column_widths

    wb = _create_workbook()
    ws = wb.active
    # Empty worksheet - no cells

    # Should not raise
    _auto_adjust_column_widths(ws, max_width=60, padding=2)
    wb.close()
