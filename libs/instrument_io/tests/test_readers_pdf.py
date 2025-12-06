"""Tests for readers.pdf module."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import PDFReadError
from instrument_io.readers.pdf import PDFReader, _is_pdf_file


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


def test_read_text_skips_empty_page_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that read_text skips pages with empty/None text.

    Covers pdf.py branch 77->75 where page_text is falsy.
    """
    from types import TracebackType
    from typing import ClassVar

    from instrument_io.readers import pdf

    class MockPage:
        def extract_text(self) -> str | None:
            return None  # Simulate empty page

    class MockPDF:
        pages: ClassVar[list[MockPage]] = [MockPage(), MockPage()]

        def __enter__(self) -> MockPDF:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            pass

    def mock_open_pdf(path: Path) -> MockPDF:
        return MockPDF()

    monkeypatch.setattr(pdf, "_open_pdf", mock_open_pdf)

    reader = PDFReader()
    test_path = Path(__file__).parent / "fixtures" / "sample.pdf"
    result = reader.read_text(test_path)

    # All pages returned None, so result should be empty
    assert result == ""


def test_read_tables_skips_empty_decoded_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that read_tables skips tables that decode to empty.

    Covers pdf.py branch 120->118 where decoded is falsy.
    """
    from types import TracebackType
    from typing import ClassVar

    from instrument_io.readers import pdf

    class MockPage:
        def extract_tables(self) -> list[list[list[str | None]]]:
            # Return a table that will decode to empty (only header row, no data)
            return [[["Header1", "Header2"]]]

    class MockPDF:
        pages: ClassVar[list[MockPage]] = [MockPage()]

        def __enter__(self) -> MockPDF:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            pass

    def mock_open_pdf(path: Path) -> MockPDF:
        return MockPDF()

    monkeypatch.setattr(pdf, "_open_pdf", mock_open_pdf)

    reader = PDFReader()
    test_path = Path(__file__).parent / "fixtures" / "sample.pdf"
    result = reader.read_tables(test_path, page_number=1)

    # Table with only headers decodes to empty list
    assert result == []
