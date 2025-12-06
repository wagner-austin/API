"""PowerPoint presentation (.pptx) reader implementation.

Provides typed reading of PowerPoint presentations via python-pptx.
Uses Protocol-based dynamic imports for external libraries.
"""

from __future__ import annotations

from pathlib import Path

from instrument_io._decoders.pptx import (
    _decode_pptx_table,
    _extract_slide_text,
    _extract_slide_title,
)
from instrument_io._exceptions import PPTXReadError
from instrument_io._protocols.python_pptx import (
    _open_pptx,
)
from instrument_io.types.common import CellValue


def _is_pptx_file(path: Path) -> bool:
    """Check if path is a PowerPoint presentation."""
    suffix = path.suffix.lower()
    return path.is_file() and suffix in (".pptx", ".pptm")


class PPTXReader:
    """Reader for PowerPoint presentations (.pptx, .pptm).

    Provides typed access to presentation text, slides, tables, and metadata.
    Uses python-pptx for extraction with Protocol-based typing for strict type safety.

    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def supports_format(self, path: Path) -> bool:
        """Check if path is a PowerPoint presentation.

        Args:
            path: Path to check.

        Returns:
            True if path is a PowerPoint presentation (.pptx or .pptm).
        """
        return _is_pptx_file(path)

    def read_text(self, path: Path) -> str:
        """Extract all text from presentation.

        Args:
            path: Path to .pptx file.

        Returns:
            Full presentation text content (all slides concatenated).

        Raises:
            PPTXReadError: If reading fails.
        """
        if not path.exists():
            raise PPTXReadError(str(path), "File does not exist")

        if not _is_pptx_file(path):
            raise PPTXReadError(str(path), "Not a PowerPoint presentation")

        prs = _open_pptx(path)
        slides_text: list[str] = []

        for slide in prs.slides:
            slide_text = _extract_slide_text(slide)
            if slide_text:
                slides_text.append(slide_text)

        return "\n\n".join(slides_text)

    def read_slides(self, path: Path) -> list[str]:
        """Extract text from each slide.

        Args:
            path: Path to .pptx file.

        Returns:
            List of slide text strings.

        Raises:
            PPTXReadError: If reading fails.
        """
        if not path.exists():
            raise PPTXReadError(str(path), "File does not exist")

        if not _is_pptx_file(path):
            raise PPTXReadError(str(path), "Not a PowerPoint presentation")

        prs = _open_pptx(path)
        result: list[str] = []

        for slide in prs.slides:
            result.append(_extract_slide_text(slide))

        return result

    def read_tables(self, path: Path) -> list[list[dict[str, CellValue]]]:
        """Extract tables from presentation.

        Args:
            path: Path to .pptx file.

        Returns:
            List of tables across all slides.
            Each table is a list of row dictionaries.
            First row of each table is used as headers.

        Raises:
            PPTXReadError: If reading fails.
        """
        if not path.exists():
            raise PPTXReadError(str(path), "File does not exist")

        if not _is_pptx_file(path):
            raise PPTXReadError(str(path), "Not a PowerPoint presentation")

        prs = _open_pptx(path)
        result: list[list[dict[str, CellValue]]] = []

        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_table:
                    table = shape.table
                    decoded = _decode_pptx_table(table)
                    if decoded:
                        result.append(decoded)

        return result

    def list_slide_titles(self, path: Path) -> list[str]:
        """List titles of all slides.

        Args:
            path: Path to .pptx file.

        Returns:
            List of slide titles.

        Raises:
            PPTXReadError: If reading fails.
        """
        if not path.exists():
            raise PPTXReadError(str(path), "File does not exist")

        if not _is_pptx_file(path):
            raise PPTXReadError(str(path), "Not a PowerPoint presentation")

        prs = _open_pptx(path)
        titles: list[str] = []

        for slide in prs.slides:
            titles.append(_extract_slide_title(slide))

        return titles

    def count_slides(self, path: Path) -> int:
        """Count number of slides in presentation.

        Args:
            path: Path to .pptx file.

        Returns:
            Number of slides.

        Raises:
            PPTXReadError: If reading fails.
        """
        if not path.exists():
            raise PPTXReadError(str(path), "File does not exist")

        if not _is_pptx_file(path):
            raise PPTXReadError(str(path), "Not a PowerPoint presentation")

        prs = _open_pptx(path)
        return len(prs.slides)


__all__ = [
    "PPTXReader",
]
