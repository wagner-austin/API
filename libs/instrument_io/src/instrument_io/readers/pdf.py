"""PDF file reader implementation.

Provides typed reading of PDF files via pdfplumber.
Uses Protocol-based dynamic imports for external libraries.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._decoders.pdf import _decode_pdf_table
from instrument_io._exceptions import PDFReadError
from instrument_io._protocols.pdfplumber import _open_pdf
from instrument_io.types.common import CellValue


def _is_pdf_file(path: Path) -> bool:
    """Check if path is a PDF file."""
    return path.is_file() and path.suffix.lower() == ".pdf"


class PDFReader:
    """Reader for PDF files via pdfplumber.

    Provides typed access to PDF text and table data. Uses pdfplumber for
    extraction with Protocol-based typing for strict type safety.

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is a PDF file.

        Args:
            path: Path to check.

        Returns:
            True if path is a PDF file (.pdf).
        """
        return _is_pdf_file(path)

    def read_text(self, path: Path, page_number: int | None = None) -> str:
        """Extract text from PDF.

        Args:
            path: Path to PDF file.
            page_number: Optional 1-based page number. If None, extracts all pages.

        Returns:
            Extracted text content.

        Raises:
            PDFReadError: If reading fails.
        """
        if not path.exists():
            raise PDFReadError(str(path), "File does not exist")

        if not _is_pdf_file(path):
            raise PDFReadError(str(path), "Not a PDF file")

        with _open_pdf(path) as pdf:
            if page_number is not None:
                # Extract from specific page
                if page_number < 1 or page_number > len(pdf.pages):
                    raise PDFReadError(
                        str(path),
                        f"Page {page_number} out of range (1-{len(pdf.pages)})",
                    )
                page = pdf.pages[page_number - 1]
                text = page.extract_text()
                return text if text else ""

            # Extract from all pages
            texts: list[str] = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texts.append(page_text)

            return "\n\n".join(texts)

    def read_tables(
        self,
        path: Path,
        page_number: int | None = None,
    ) -> list[list[dict[str, CellValue]]]:
        """Extract tables from PDF.

        Args:
            path: Path to PDF file.
            page_number: Optional 1-based page number. If None, extracts from all pages.

        Returns:
            List of tables, where each table is a list of row dictionaries.
            First row of each table is used as headers.

        Raises:
            PDFReadError: If reading fails.
        """
        if not path.exists():
            raise PDFReadError(str(path), "File does not exist")

        if not _is_pdf_file(path):
            raise PDFReadError(str(path), "Not a PDF file")

        with _open_pdf(path) as pdf:
            result: list[list[dict[str, CellValue]]] = []

            if page_number is not None:
                # Extract from specific page
                if page_number < 1 or page_number > len(pdf.pages):
                    raise PDFReadError(
                        str(path),
                        f"Page {page_number} out of range (1-{len(pdf.pages)})",
                    )
                page = pdf.pages[page_number - 1]
                tables = page.extract_tables()
                for table in tables:
                    decoded = _decode_pdf_table(table)
                    if decoded:
                        result.append(decoded)
                return result

            # Extract from all pages
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    decoded = _decode_pdf_table(table)
                    if decoded:
                        result.append(decoded)

            return result

    def count_pages(self, path: Path) -> int:
        """Count number of pages in PDF.

        Args:
            path: Path to PDF file.

        Returns:
            Number of pages.

        Raises:
            PDFReadError: If reading fails.
        """
        if not path.exists():
            raise PDFReadError(str(path), "File does not exist")

        if not _is_pdf_file(path):
            raise PDFReadError(str(path), "Not a PDF file")

        with _open_pdf(path) as pdf:
            return len(pdf.pages)


__all__ = [
    "PDFReader",
]
