"""Tests for _protocols.python_docx module."""

from __future__ import annotations

from pathlib import Path

from instrument_io._protocols.python_docx import (
    _create_document,
    _get_pt,
    _get_wd_align_center,
)


def test_create_document() -> None:
    """Test creating a new Word document."""
    doc = _create_document()

    # Verify we can add content to the document
    doc.add_heading("Test Title", level=0)
    doc.add_paragraph("Test paragraph")

    # Document should have content - at least title and paragraph
    assert len(doc.paragraphs) >= 2


def test_create_document_and_save(tmp_path: Path) -> None:
    """Test creating and saving a Word document."""
    doc = _create_document()
    doc.add_heading("Test Document", level=0)
    doc.add_paragraph("This is a test paragraph.")

    output_path = tmp_path / "test.docx"
    doc.save(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_get_wd_align_center() -> None:
    """Test getting center alignment value."""
    center = _get_wd_align_center()

    # The actual value is 1 for CENTER alignment
    assert center == 1


def test_document_with_alignment(tmp_path: Path) -> None:
    """Test creating document with centered alignment."""
    doc = _create_document()
    center = _get_wd_align_center()

    heading = doc.add_heading("Centered Title", level=0)
    heading.alignment = center

    paragraph = doc.add_paragraph("Centered paragraph")
    paragraph.alignment = center

    output_path = tmp_path / "aligned.docx"
    doc.save(output_path)

    assert output_path.exists()


def test_document_add_table(tmp_path: Path) -> None:
    """Test creating document with table."""
    doc = _create_document()

    table = doc.add_table(rows=2, cols=3)
    table.style = "Table Grid"

    # Fill header row
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Column 1"
    hdr_cells[1].text = "Column 2"
    hdr_cells[2].text = "Column 3"

    # Fill data row
    row_cells = table.add_row().cells
    row_cells[0].text = "Data 1"
    row_cells[1].text = "Data 2"
    row_cells[2].text = "Data 3"

    output_path = tmp_path / "table.docx"
    doc.save(output_path)

    assert output_path.exists()


def test_get_pt() -> None:
    """Test getting Pt (points) length object."""
    pt_value = _get_pt(12.0)

    # Pt is defined as 914400 EMUs per inch, 72 points per inch
    # So 1 point = 914400 / 72 = 12700 EMUs
    # 12 points = 12 * 12700 = 152400 EMUs
    expected_emu = 12 * 12700
    assert pt_value.emu == expected_emu
