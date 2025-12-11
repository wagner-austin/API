"""Tests for readers.pdf module."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import PDFReadError
from instrument_io.readers.pdf import PDFReader, _is_pdf_file
from instrument_io.testing import FakePDF, FakePDFPage, hooks


def test_is_pdf_file_valid(tmp_path: Path) -> None:
    file = tmp_path / "test.pdf"
    file.touch()
    assert _is_pdf_file(file) is True


def test_is_pdf_file_uppercase_extension(tmp_path: Path) -> None:
    file = tmp_path / "test.PDF"
    file.touch()
    assert _is_pdf_file(file) is True


def test_is_pdf_file_wrong_extension(tmp_path: Path) -> None:
    file = tmp_path / "test.txt"
    file.touch()
    assert _is_pdf_file(file) is False


def test_is_pdf_file_directory(tmp_path: Path) -> None:
    directory = tmp_path / "test.pdf"
    directory.mkdir()
    assert _is_pdf_file(directory) is False


def test_is_pdf_file_not_exists(tmp_path: Path) -> None:
    file = tmp_path / "nonexistent.pdf"
    assert _is_pdf_file(file) is False


def test_pdf_reader_supports_format_valid(tmp_path: Path) -> None:
    reader = PDFReader()
    file = tmp_path / "test.pdf"
    file.touch()
    assert reader.supports_format(file) is True


def test_pdf_reader_supports_format_invalid(tmp_path: Path) -> None:
    reader = PDFReader()
    file = tmp_path / "test.txt"
    file.touch()
    assert reader.supports_format(file) is False


def test_pdf_reader_read_text_file_not_exists(tmp_path: Path) -> None:
    reader = PDFReader()
    file = tmp_path / "nonexistent.pdf"
    with pytest.raises(PDFReadError) as exc_info:
        reader.read_text(file)
    assert "File does not exist" in str(exc_info.value)


def test_pdf_reader_read_text_not_pdf(tmp_path: Path) -> None:
    reader = PDFReader()
    file = tmp_path / "test.txt"
    file.write_text("not a pdf")
    with pytest.raises(PDFReadError) as exc_info:
        reader.read_text(file)
    assert "Not a PDF file" in str(exc_info.value)


def test_pdf_reader_read_tables_file_not_exists(tmp_path: Path) -> None:
    reader = PDFReader()
    file = tmp_path / "nonexistent.pdf"
    with pytest.raises(PDFReadError) as exc_info:
        reader.read_tables(file)
    assert "File does not exist" in str(exc_info.value)


def test_pdf_reader_read_tables_not_pdf(tmp_path: Path) -> None:
    reader = PDFReader()
    file = tmp_path / "test.txt"
    file.write_text("not a pdf")
    with pytest.raises(PDFReadError) as exc_info:
        reader.read_tables(file)
    assert "Not a PDF file" in str(exc_info.value)


def test_pdf_reader_count_pages_file_not_exists(tmp_path: Path) -> None:
    reader = PDFReader()
    file = tmp_path / "nonexistent.pdf"
    with pytest.raises(PDFReadError) as exc_info:
        reader.count_pages(file)
    assert "File does not exist" in str(exc_info.value)


def test_pdf_reader_count_pages_not_pdf(tmp_path: Path) -> None:
    reader = PDFReader()
    file = tmp_path / "test.txt"
    file.write_text("not a pdf")
    with pytest.raises(PDFReadError) as exc_info:
        reader.count_pages(file)
    assert "Not a PDF file" in str(exc_info.value)


class TestPDFReaderReadText:
    """Tests for PDFReader.read_text method using hooks."""

    def test_read_text_all_pages(self, tmp_path: Path) -> None:
        """Test reading text from all pages."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF(
            [
                FakePDFPage(text="Page 1 content"),
                FakePDFPage(text="Page 2 content"),
            ]
        )
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        result = reader.read_text(file)

        assert result == "Page 1 content\n\nPage 2 content"

    def test_read_text_specific_page(self, tmp_path: Path) -> None:
        """Test reading text from a specific page."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF(
            [
                FakePDFPage(text="Page 1 content"),
                FakePDFPage(text="Page 2 content"),
            ]
        )
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        result = reader.read_text(file, page_number=2)

        assert result == "Page 2 content"

    def test_read_text_page_out_of_range(self, tmp_path: Path) -> None:
        """Test error when page number is out of range."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF([FakePDFPage(text="Page 1")])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        with pytest.raises(PDFReadError) as exc_info:
            reader.read_text(file, page_number=5)
        assert "Page 5 out of range" in str(exc_info.value)

    def test_read_text_page_zero_out_of_range(self, tmp_path: Path) -> None:
        """Test error when page number is 0 (pages are 1-based)."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF([FakePDFPage(text="Page 1")])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        with pytest.raises(PDFReadError) as exc_info:
            reader.read_text(file, page_number=0)
        assert "Page 0 out of range" in str(exc_info.value)

    def test_read_text_skips_empty_page_text(self, tmp_path: Path) -> None:
        """Test that read_text skips pages with empty/None text.

        Covers pdf.py branch 77->75 where page_text is falsy.
        """
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF(
            [
                FakePDFPage(text=None),  # Empty page
                FakePDFPage(text=None),  # Empty page
            ]
        )
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        result = reader.read_text(file)

        # All pages returned None, so result should be empty
        assert result == ""

    def test_read_text_empty_page_returns_empty_string(self, tmp_path: Path) -> None:
        """Test that specific page with None text returns empty string."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF([FakePDFPage(text=None)])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        result = reader.read_text(file, page_number=1)

        assert result == ""


class TestPDFReaderReadTables:
    """Tests for PDFReader.read_tables method using hooks."""

    def test_read_tables_all_pages(self, tmp_path: Path) -> None:
        """Test reading tables from all pages."""
        file = tmp_path / "test.pdf"
        file.touch()

        # Table with header + data row
        table_data: list[list[str | None]] = [
            ["Name", "Value"],
            ["Test", "123"],
        ]
        fake_pdf = FakePDF([FakePDFPage(tables=[table_data])])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        result = reader.read_tables(file)

        assert len(result) == 1
        # Decoder converts numeric strings to int
        assert result[0] == [{"Name": "Test", "Value": 123}]

    def test_read_tables_specific_page(self, tmp_path: Path) -> None:
        """Test reading tables from a specific page."""
        file = tmp_path / "test.pdf"
        file.touch()

        table1: list[list[str | None]] = [["A", "B"], ["1", "2"]]
        table2: list[list[str | None]] = [["X", "Y"], ["3", "4"]]
        fake_pdf = FakePDF(
            [
                FakePDFPage(tables=[table1]),
                FakePDFPage(tables=[table2]),
            ]
        )
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        result = reader.read_tables(file, page_number=2)

        assert len(result) == 1
        # Decoder converts numeric strings to int
        assert result[0] == [{"X": 3, "Y": 4}]

    def test_read_tables_page_out_of_range(self, tmp_path: Path) -> None:
        """Test error when page number is out of range for tables."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF([FakePDFPage(tables=[])])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        with pytest.raises(PDFReadError) as exc_info:
            reader.read_tables(file, page_number=10)
        assert "Page 10 out of range" in str(exc_info.value)

    def test_read_tables_skips_empty_decoded_tables(self, tmp_path: Path) -> None:
        """Test that read_tables skips tables that decode to empty.

        Covers pdf.py branch 120->118 where decoded is falsy.
        """
        file = tmp_path / "test.pdf"
        file.touch()

        # Table with only header row (no data) decodes to empty
        table_data: list[list[str | None]] = [["Header1", "Header2"]]
        fake_pdf = FakePDF([FakePDFPage(tables=[table_data])])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        result = reader.read_tables(file, page_number=1)

        # Table with only headers decodes to empty list
        assert result == []

    def test_read_tables_skips_empty_on_all_pages(self, tmp_path: Path) -> None:
        """Test that read_tables skips empty tables across all pages."""
        file = tmp_path / "test.pdf"
        file.touch()

        # Tables with only header rows
        table1: list[list[str | None]] = [["A"]]
        table2: list[list[str | None]] = [["B"]]
        fake_pdf = FakePDF(
            [
                FakePDFPage(tables=[table1]),
                FakePDFPage(tables=[table2]),
            ]
        )
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        result = reader.read_tables(file)

        assert result == []


class TestPDFReaderCountPages:
    """Tests for PDFReader.count_pages method using hooks."""

    def test_count_pages_single(self, tmp_path: Path) -> None:
        """Test counting single page PDF."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF([FakePDFPage()])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        count = reader.count_pages(file)

        assert count == 1

    def test_count_pages_multiple(self, tmp_path: Path) -> None:
        """Test counting multi-page PDF."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF([FakePDFPage(), FakePDFPage(), FakePDFPage()])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        count = reader.count_pages(file)

        assert count == 3

    def test_count_pages_empty(self, tmp_path: Path) -> None:
        """Test counting PDF with no pages."""
        file = tmp_path / "test.pdf"
        file.touch()

        fake_pdf = FakePDF([])
        hooks.open_pdf = lambda p: fake_pdf

        reader = PDFReader()
        count = reader.count_pages(file)

        assert count == 0
