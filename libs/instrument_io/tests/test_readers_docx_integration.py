"""Integration tests for DOCX reader using actual files."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import DOCXReadError
from instrument_io.readers.docx import DOCXReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DOCX = FIXTURES_DIR / "sample.docx"
COMPLEX_DOCX = FIXTURES_DIR / "complex.docx"
EDGE_CASES_DOCX = FIXTURES_DIR / "edge_cases.docx"
FLOAT_EDGE_CASES_DOCX = FIXTURES_DIR / "float_edge_cases.docx"
EMPTY_TABLE_DOCX = FIXTURES_DIR / "empty_table.docx"
EMPTY_HEADERS_DOCX = FIXTURES_DIR / "empty_headers.docx"
ALL_EMPTY_HEADERS_DOCX = FIXTURES_DIR / "all_empty_headers.docx"
CUSTOM_HEADING_DOCX = FIXTURES_DIR / "custom_heading.docx"


def test_read_text() -> None:
    """Test reading all text from actual DOCX file."""
    reader = DOCXReader()
    text = reader.read_text(SAMPLE_DOCX)

    assert "Test Document" in text
    assert "test paragraph" in text
    assert "Section 2" in text


def test_read_paragraphs() -> None:
    """Test reading paragraphs from actual DOCX file."""
    reader = DOCXReader()
    paragraphs = reader.read_paragraphs(SAMPLE_DOCX)

    # Filter out empty paragraphs
    non_empty = [p for p in paragraphs if p.strip()]
    assert non_empty  # Ensure we have paragraphs
    assert any("Test Document" in p for p in non_empty)
    assert any("test paragraph" in p for p in non_empty)


def test_read_tables() -> None:
    """Test reading tables from actual DOCX file."""
    reader = DOCXReader()
    tables = reader.read_tables(SAMPLE_DOCX)

    assert len(tables) == 1
    table = tables[0]

    # Should have 3 rows (including header)
    assert len(table) == 2  # Data rows only (header row used for keys)

    # Check first data row
    assert table[0]["Name"] == "Alice"
    assert table[0]["Age"] == 25
    assert table[0]["Score"] == 95.5

    # Check second data row
    assert table[1]["Name"] == "Bob"
    assert table[1]["Age"] == 30
    assert table[1]["Score"] == 87.3


def test_read_headings() -> None:
    """Test reading headings from actual DOCX file."""
    reader = DOCXReader()
    headings = reader.read_headings(SAMPLE_DOCX)

    assert len(headings) >= 2
    # Headings are (level, text) tuples
    assert any(h[1] == "Test Document" and h[0] == 1 for h in headings)
    assert any(h[1] == "Section 2" and h[0] == 2 for h in headings)


def test_supports_format_docx() -> None:
    """Test format detection for .docx files."""
    reader = DOCXReader()
    assert reader.supports_format(SAMPLE_DOCX) is True


def test_supports_format_non_docx() -> None:
    """Test format detection for non-.docx files."""
    reader = DOCXReader()
    non_docx = FIXTURES_DIR / "sample.txt"
    assert reader.supports_format(non_docx) is False


def test_read_text_file_not_exists() -> None:
    """Test error when file doesn't exist."""
    reader = DOCXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.docx"

    with pytest.raises(DOCXReadError) as exc_info:
        reader.read_text(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_text_not_docx_file() -> None:
    """Test error when file is not a .docx file."""
    reader = DOCXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(DOCXReadError) as exc_info:
        reader.read_text(txt_file)

    assert "Not a Word document" in str(exc_info.value)


def test_read_paragraphs_file_not_exists() -> None:
    """Test error when file doesn't exist for read_paragraphs."""
    reader = DOCXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.docx"

    with pytest.raises(DOCXReadError) as exc_info:
        reader.read_paragraphs(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_paragraphs_not_docx_file() -> None:
    """Test error when file is not a .docx file for read_paragraphs."""
    reader = DOCXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(DOCXReadError) as exc_info:
        reader.read_paragraphs(txt_file)

    assert "Not a Word document" in str(exc_info.value)


def test_read_tables_file_not_exists() -> None:
    """Test error when file doesn't exist for read_tables."""
    reader = DOCXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.docx"

    with pytest.raises(DOCXReadError) as exc_info:
        reader.read_tables(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_tables_not_docx_file() -> None:
    """Test error when file is not a .docx file for read_tables."""
    reader = DOCXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(DOCXReadError) as exc_info:
        reader.read_tables(txt_file)

    assert "Not a Word document" in str(exc_info.value)


def test_read_headings_file_not_exists() -> None:
    """Test error when file doesn't exist for read_headings."""
    reader = DOCXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.docx"

    with pytest.raises(DOCXReadError) as exc_info:
        reader.read_headings(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_headings_not_docx_file() -> None:
    """Test error when file is not a .docx file for read_headings."""
    reader = DOCXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(DOCXReadError) as exc_info:
        reader.read_headings(txt_file)

    assert "Not a Word document" in str(exc_info.value)


def test_read_complex_docx() -> None:
    """Test reading complex real DOCX file from Faiola Lab."""
    reader = DOCXReader()

    # Test reading text from complex file
    text = reader.read_text(COMPLEX_DOCX)
    assert len(text) > 100  # Should have substantial content

    # Test reading paragraphs
    paragraphs = reader.read_paragraphs(COMPLEX_DOCX)
    assert len(paragraphs) > 5  # Should have multiple paragraphs

    # Test reading tables (may or may not have tables)
    tables = reader.read_tables(COMPLEX_DOCX)
    assert tables == tables  # Verify it doesn't crash

    # Test reading headings
    headings = reader.read_headings(COMPLEX_DOCX)
    assert headings == headings  # Verify it doesn't crash


def test_read_edge_cases_docx() -> None:
    """Test reading DOCX with edge case values (booleans, negatives, empty, etc)."""
    reader = DOCXReader()

    # Read tables with edge cases
    tables = reader.read_tables(EDGE_CASES_DOCX)
    assert len(tables) == 1

    table = tables[0]
    assert len(table) == 7  # 7 data rows (excluding header)

    # Check boolean true values
    assert table[0]["Value"] is True
    assert table[0]["Expected"] is True

    # Check boolean false values
    assert table[1]["Value"] is False
    assert table[1]["Expected"] is False

    # Check negative integers
    assert table[2]["Value"] == -42
    assert table[2]["Expected"] == -100

    # Check floats
    assert table[3]["Value"] == 3.14
    assert table[3]["Expected"] == -2.71

    # Check scientific notation
    assert table[4]["Value"] == 1.5e10
    assert table[4]["Expected"] == -2.3e-5

    # Check empty/whitespace values
    assert table[5]["Value"] is None
    assert table[5]["Expected"] is None

    # Check strings
    assert table[6]["Value"] == "test"
    assert table[6]["Expected"] == "hello world"


def test_read_float_edge_cases_docx() -> None:
    """Test reading DOCX with float validation edge cases."""
    reader = DOCXReader()

    # Read tables with float edge cases
    tables = reader.read_tables(FLOAT_EDGE_CASES_DOCX)
    assert len(tables) == 1

    table = tables[0]
    assert len(table) == 5  # 5 data rows (excluding header)

    # All should be treated as strings since they're invalid floats
    # Empty string becomes None
    assert table[0]["Test"] is None or table[0]["Test"] == ""
    assert table[0]["Expected"] is False

    # Just "+" becomes a string
    assert table[1]["Test"] == "+"
    assert table[1]["Expected"] is False

    # Just "-" becomes a string
    assert table[2]["Test"] == "-"
    assert table[2]["Expected"] is False

    # Invalid decimal "1..2" becomes a string
    assert table[3]["Test"] == "1..2"
    assert table[3]["Expected"] is False

    # Invalid decimal "1.2.3" becomes a string
    assert table[4]["Test"] == "1.2.3"
    assert table[4]["Expected"] is False


def test_read_empty_table_docx() -> None:
    """Test reading DOCX with empty table (header row only, no data rows).

    This covers the decoder branch where table has rows but no data rows.
    The reader filters out empty tables, so we expect 0 tables in result.
    The decoder returns [] which exercises the empty table code path.
    """
    reader = DOCXReader()

    tables = reader.read_tables(EMPTY_TABLE_DOCX)
    # Empty tables are filtered out by the reader
    assert len(tables) == 0


def test_read_empty_headers_docx() -> None:
    """Test reading DOCX with table that has empty header cells.

    This covers the decoder branch where some headers are empty and skipped.
    """
    reader = DOCXReader()

    tables = reader.read_tables(EMPTY_HEADERS_DOCX)
    assert len(tables) == 1

    table = tables[0]
    assert len(table) == 2  # Two data rows

    # Only non-empty headers should be present in the result
    assert "Name" in table[0]
    assert "Value" in table[0]
    # Empty headers should not be present
    assert "" not in table[0]

    # Check values
    assert table[0]["Name"] == "Alice"
    assert table[0]["Value"] == 100
    assert table[1]["Name"] == "Bob"
    assert table[1]["Value"] == 200


def test_read_all_empty_headers_docx() -> None:
    """Test reading DOCX with table where ALL headers are empty.

    This covers the decoder branch where row_dict is empty after processing.
    The reader filters out tables with no extracted data.
    """
    reader = DOCXReader()

    tables = reader.read_tables(ALL_EMPTY_HEADERS_DOCX)
    # Tables with all empty headers produce no data and are filtered out
    assert len(tables) == 0


def test_read_custom_heading_style_docx() -> None:
    """Test reading DOCX with custom heading style that has non-digit name.

    This covers the decoder branch where a style starts with "Heading "
    but the second part is not a digit (e.g., "Heading Overview").
    Such styles should not be recognized as valid heading levels.
    """
    reader = DOCXReader()

    headings = reader.read_headings(CUSTOM_HEADING_DOCX)

    # Only the valid "Heading 1" should be recognized
    # "Heading Overview" should be filtered out (returns None for level)
    assert len(headings) == 1
    assert headings[0][0] == 1  # Level 1
    assert headings[0][1] == "Valid Heading"
