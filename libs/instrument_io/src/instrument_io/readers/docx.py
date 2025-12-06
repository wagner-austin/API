"""Word document (.docx) reader implementation.

Provides typed reading of Word documents via python-docx.
Uses Protocol-based dynamic imports for external libraries.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._decoders.docx import (
    _decode_docx_table,
    _decode_paragraph_text,
    _get_heading_level,
)
from instrument_io._exceptions import DOCXReadError
from instrument_io._protocols.python_docx import _open_docx
from instrument_io.types.common import CellValue


def _is_docx_file(path: Path) -> bool:
    """Check if path is a Word document."""
    return path.is_file() and path.suffix.lower() == ".docx"


class DOCXReader:
    """Reader for Word documents (.docx).

    Provides typed access to document text, paragraphs, tables, and headings.
    Uses python-docx for extraction with Protocol-based typing for strict type safety.

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is a Word document.

        Args:
            path: Path to check.

        Returns:
            True if path is a Word document (.docx).
        """
        return _is_docx_file(path)

    def read_text(self, path: Path) -> str:
        """Extract all text from document.

        Args:
            path: Path to .docx file.

        Returns:
            Full document text content.

        Raises:
            DOCXReadError: If reading fails.
        """
        if not path.exists():
            raise DOCXReadError(str(path), "File does not exist")

        if not _is_docx_file(path):
            raise DOCXReadError(str(path), "Not a Word document")

        doc = _open_docx(path)
        paragraphs_text = [_decode_paragraph_text(p) for p in doc.paragraphs]
        return "\n".join(paragraphs_text)

    def read_paragraphs(self, path: Path) -> list[str]:
        """Extract paragraphs from document.

        Args:
            path: Path to .docx file.

        Returns:
            List of paragraph text strings.

        Raises:
            DOCXReadError: If reading fails.
        """
        if not path.exists():
            raise DOCXReadError(str(path), "File does not exist")

        if not _is_docx_file(path):
            raise DOCXReadError(str(path), "Not a Word document")

        doc = _open_docx(path)
        return [_decode_paragraph_text(p) for p in doc.paragraphs]

    def read_tables(self, path: Path) -> list[list[dict[str, CellValue]]]:
        """Extract tables from document.

        Args:
            path: Path to .docx file.

        Returns:
            List of tables, where each table is a list of row dictionaries.
            First row of each table is used as headers.

        Raises:
            DOCXReadError: If reading fails.
        """
        if not path.exists():
            raise DOCXReadError(str(path), "File does not exist")

        if not _is_docx_file(path):
            raise DOCXReadError(str(path), "Not a Word document")

        doc = _open_docx(path)
        result: list[list[dict[str, CellValue]]] = []
        for table in doc.tables:
            decoded = _decode_docx_table(table)
            if decoded:
                result.append(decoded)
        return result

    def read_headings(self, path: Path) -> list[tuple[int, str]]:
        """Extract headings from document.

        Args:
            path: Path to .docx file.

        Returns:
            List of (level, text) tuples for each heading.
            Level is 1-9 for heading styles.

        Raises:
            DOCXReadError: If reading fails.
        """
        if not path.exists():
            raise DOCXReadError(str(path), "File does not exist")

        if not _is_docx_file(path):
            raise DOCXReadError(str(path), "Not a Word document")

        doc = _open_docx(path)
        headings: list[tuple[int, str]] = []
        for para in doc.paragraphs:
            level = _get_heading_level(para)
            if level is not None:
                headings.append((level, _decode_paragraph_text(para)))
        return headings


__all__ = [
    "DOCXReader",
]
