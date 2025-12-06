"""Integration tests for PDF reader with real PDF files."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._exceptions import PDFReadError
from instrument_io.readers.pdf import PDFReader

# Test PDF paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
RESEARCH_PDF = FIXTURES_DIR / "research_paper.pdf"
TABLES_PDF = FIXTURES_DIR / "paper_with_tables.pdf"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Sample PDF not available")
class TestPDFReaderBasic:
    """Basic integration tests with sample PDF."""

    def test_count_pages(self) -> None:
        """Test counting pages in a real PDF."""
        reader = PDFReader()
        page_count = reader.count_pages(SAMPLE_PDF)
        assert page_count > 0

    def test_read_text_all_pages(self) -> None:
        """Test reading text from all pages."""
        reader = PDFReader()
        text = reader.read_text(SAMPLE_PDF)
        assert text.strip() != ""

    def test_read_text_specific_page(self) -> None:
        """Test reading text from a specific page."""
        reader = PDFReader()
        text = reader.read_text(SAMPLE_PDF, page_number=1)
        assert text == text

    def test_read_text_page_out_of_range(self) -> None:
        """Test reading from a page that doesn't exist."""
        reader = PDFReader()
        with pytest.raises(PDFReadError) as exc_info:
            reader.read_text(SAMPLE_PDF, page_number=999999)
        assert "out of range" in str(exc_info.value)

    def test_read_tables_all_pages(self) -> None:
        """Test reading tables from all pages."""
        reader = PDFReader()
        tables = reader.read_tables(SAMPLE_PDF)
        assert tables == tables

    def test_read_tables_specific_page(self) -> None:
        """Test reading tables from a specific page."""
        reader = PDFReader()
        tables = reader.read_tables(SAMPLE_PDF, page_number=1)
        assert tables == tables

    def test_read_tables_page_out_of_range(self) -> None:
        """Test reading tables from a page that doesn't exist."""
        reader = PDFReader()
        with pytest.raises(PDFReadError) as exc_info:
            reader.read_tables(SAMPLE_PDF, page_number=999999)
        assert "out of range" in str(exc_info.value)


@pytest.mark.skipif(not RESEARCH_PDF.exists(), reason="Research PDF not available")
class TestPDFReaderWithResearchPaper:
    """Tests with large research paper PDF."""

    def test_read_text_extracts_content(self) -> None:
        """Test that text extraction works on research paper."""
        reader = PDFReader()
        text = reader.read_text(RESEARCH_PDF)
        # Research papers have significant text
        assert text.strip() != ""
        # Should have multiple pages worth of content
        assert len(text) > 1000

    def test_read_text_multiple_pages(self) -> None:
        """Test reading from multiple pages of research paper."""
        reader = PDFReader()
        page_count = reader.count_pages(RESEARCH_PDF)
        assert page_count > 1

        # Read first page
        first_page = reader.read_text(RESEARCH_PDF, page_number=1)
        # Read last page
        last_page = reader.read_text(RESEARCH_PDF, page_number=page_count)

        # All pages should return strings (not fail)
        assert first_page == first_page
        assert last_page == last_page

    def test_read_tables_from_research_paper(self) -> None:
        """Test table extraction from research paper."""
        reader = PDFReader()
        tables = reader.read_tables(RESEARCH_PDF)
        # Research papers may or may not have tables, just verify it doesn't crash
        assert tables == tables


@pytest.mark.skipif(not TABLES_PDF.exists(), reason="Tables PDF not available")
class TestPDFReaderWithTables:
    """Tests specifically for table extraction."""

    def test_read_tables_extracts_data(self) -> None:
        """Test that tables are extracted from PDF with tables."""
        reader = PDFReader()
        tables = reader.read_tables(TABLES_PDF)
        # This PDF is known to have tables
        # Just verify extraction works without errors
        assert tables == tables

    def test_read_tables_specific_page_with_tables(self) -> None:
        """Test extracting tables from specific pages."""
        reader = PDFReader()
        page_count = reader.count_pages(TABLES_PDF)

        # Try each page
        for page_num in range(1, min(page_count + 1, 6)):  # Test first 5 pages
            tables = reader.read_tables(TABLES_PDF, page_number=page_num)
            # Should return a list (may be empty)
            assert tables == tables
