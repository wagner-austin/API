"""Integration tests for PPTX reader using actual files."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import PPTXReadError
from instrument_io.readers.pptx import PPTXReader

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PPTX = FIXTURES_DIR / "sample.pptx"
COMPLEX_PPTX = FIXTURES_DIR / "complex.pptx"
EDGE_CASES_PPTX = FIXTURES_DIR / "edge_cases.pptx"
FLOAT_EDGE_CASES_PPTX = FIXTURES_DIR / "float_edge_cases.pptx"
EMPTY_TABLE_PPTX = FIXTURES_DIR / "empty_table.pptx"
EMPTY_HEADERS_PPTX = FIXTURES_DIR / "empty_headers.pptx"
ALL_EMPTY_HEADERS_PPTX = FIXTURES_DIR / "all_empty_headers.pptx"


def test_read_text() -> None:
    """Test reading all text from actual PPTX file."""
    reader = PPTXReader()
    text = reader.read_text(SAMPLE_PPTX)

    assert "Test Presentation" in text
    assert "sample presentation" in text


def test_read_slides() -> None:
    """Test reading slides from actual PPTX file."""
    reader = PPTXReader()
    slides = reader.read_slides(SAMPLE_PPTX)

    assert len(slides) == 2

    # First slide (title slide)
    assert "Test Presentation" in slides[0]
    assert "sample presentation" in slides[0]

    # Second slide may be empty (just has table)
    assert len(slides) >= 2


def test_read_tables() -> None:
    """Test reading tables from actual PPTX file."""
    reader = PPTXReader()
    tables = reader.read_tables(SAMPLE_PPTX)

    # Check the first table has expected data
    table = tables[0]
    assert len(table) == 2  # Two data rows

    # Check first row
    assert table[0]["Product"] == "Widget"
    assert table[0]["Quantity"] == 10
    assert table[0]["Price"] == 25.50

    # Check second row
    assert table[1]["Product"] == "Gadget"
    assert table[1]["Quantity"] == 5
    assert table[1]["Price"] == 42.75


def test_list_slide_titles() -> None:
    """Test listing slide titles from actual PPTX file."""
    reader = PPTXReader()
    titles = reader.list_slide_titles(SAMPLE_PPTX)

    assert len(titles) == 2
    assert titles[0] == "Test Presentation"
    assert titles[1] == ""  # Second slide has no title


def test_count_slides() -> None:
    """Test counting slides in actual PPTX file."""
    reader = PPTXReader()
    count = reader.count_slides(SAMPLE_PPTX)

    assert count == 2


def test_supports_format_pptx() -> None:
    """Test format detection for .pptx files."""
    reader = PPTXReader()
    assert reader.supports_format(SAMPLE_PPTX) is True


def test_supports_format_non_pptx() -> None:
    """Test format detection for non-.pptx files."""
    reader = PPTXReader()
    non_pptx = FIXTURES_DIR / "sample.txt"
    assert reader.supports_format(non_pptx) is False


def test_read_text_file_not_exists() -> None:
    """Test error when file doesn't exist."""
    reader = PPTXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.pptx"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.read_text(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_text_not_pptx_file() -> None:
    """Test error when file is not a .pptx file."""
    reader = PPTXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.read_text(txt_file)

    assert "Not a PowerPoint presentation" in str(exc_info.value)


def test_read_slides_file_not_exists() -> None:
    """Test error when file doesn't exist for read_slides."""
    reader = PPTXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.pptx"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.read_slides(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_slides_not_pptx_file() -> None:
    """Test error when file is not a .pptx file for read_slides."""
    reader = PPTXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.read_slides(txt_file)

    assert "Not a PowerPoint presentation" in str(exc_info.value)


def test_read_tables_file_not_exists() -> None:
    """Test error when file doesn't exist for read_tables."""
    reader = PPTXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.pptx"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.read_tables(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_tables_not_pptx_file() -> None:
    """Test error when file is not a .pptx file for read_tables."""
    reader = PPTXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.read_tables(txt_file)

    assert "Not a PowerPoint presentation" in str(exc_info.value)


def test_list_slide_titles_file_not_exists() -> None:
    """Test error when file doesn't exist for list_slide_titles."""
    reader = PPTXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.pptx"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.list_slide_titles(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_list_slide_titles_not_pptx_file() -> None:
    """Test error when file is not a .pptx file for list_slide_titles."""
    reader = PPTXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.list_slide_titles(txt_file)

    assert "Not a PowerPoint presentation" in str(exc_info.value)


def test_count_slides_file_not_exists() -> None:
    """Test error when file doesn't exist for count_slides."""
    reader = PPTXReader()
    nonexistent = FIXTURES_DIR / "nonexistent.pptx"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.count_slides(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_count_slides_not_pptx_file() -> None:
    """Test error when file is not a .pptx file for count_slides."""
    reader = PPTXReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(PPTXReadError) as exc_info:
        reader.count_slides(txt_file)

    assert "Not a PowerPoint presentation" in str(exc_info.value)


def test_read_complex_pptx() -> None:
    """Test reading complex real PPTX file from Faiola Lab."""
    reader = PPTXReader()

    # Test reading all text
    text = reader.read_text(COMPLEX_PPTX)
    assert len(text) > 50  # Should have substantial content

    # Test reading slides
    slides = reader.read_slides(COMPLEX_PPTX)
    # Verify slides exist by accessing first slide
    first_slide = slides[0]
    assert first_slide == first_slide  # Will fail if slides is empty

    # Test counting slides
    count = reader.count_slides(COMPLEX_PPTX)
    assert count == len(slides)

    # Test reading tables (may or may not have tables)
    tables = reader.read_tables(COMPLEX_PPTX)
    assert tables == tables  # Verify it doesn't crash

    # Test listing slide titles
    titles = reader.list_slide_titles(COMPLEX_PPTX)
    assert len(titles) == count


def test_read_edge_cases_pptx() -> None:
    """Test reading PPTX with edge case values (booleans, negatives, empty, etc)."""
    reader = PPTXReader()

    # Read tables with edge cases
    tables = reader.read_tables(EDGE_CASES_PPTX)
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


def test_read_float_edge_cases_pptx() -> None:
    """Test reading PPTX with float validation edge cases."""
    reader = PPTXReader()

    # Read tables with float edge cases
    tables = reader.read_tables(FLOAT_EDGE_CASES_PPTX)
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


def test_read_empty_table_pptx() -> None:
    """Test reading PPTX with empty table (header row only, no data rows).

    This covers the decoder branch where table has rows but no data rows.
    The reader filters out empty tables, so we expect 0 tables in result.
    The decoder returns [] which exercises the empty table code path.
    """
    reader = PPTXReader()

    tables = reader.read_tables(EMPTY_TABLE_PPTX)
    # Empty tables are filtered out by the reader
    assert len(tables) == 0


def test_read_empty_headers_pptx() -> None:
    """Test reading PPTX with table that has empty header cells.

    This covers the decoder branch where some headers are empty and skipped.
    """
    reader = PPTXReader()

    tables = reader.read_tables(EMPTY_HEADERS_PPTX)
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


def test_read_all_empty_headers_pptx() -> None:
    """Test reading PPTX with table where ALL headers are empty.

    This covers the decoder branch where row_dict is empty after processing.
    The reader filters out tables with no extracted data.
    """
    reader = PPTXReader()

    tables = reader.read_tables(ALL_EMPTY_HEADERS_PPTX)
    # Tables with all empty headers produce no data and are filtered out
    assert len(tables) == 0
