"""Protocol definitions for writer interfaces.

Defines the typed contracts that writer implementations must fulfill.
No recovery, no best-effort - failures propagate as exceptions.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from instrument_io.types.common import CellValue
from instrument_io.types.document import DocumentContent


class DocumentWriterProtocol(Protocol):
    """Protocol for writing document files (Word, PDF).

    Implementations must provide a method for writing structured
    document content. All methods raise exceptions on failure.
    """

    def write_document(
        self,
        content: DocumentContent,
        out_path: Path,
    ) -> None:
        """Write content to a document file.

        Args:
            content: List of document sections to write.
            out_path: Output file path.

        Raises:
            WriterError: If writing fails.
        """
        ...


class ExcelWriterProtocol(Protocol):
    """Protocol for writing Excel files.

    Implementations must provide methods for writing typed data to Excel.
    All methods raise exceptions on failure - no recovery or fallbacks.
    """

    def write_sheet(
        self,
        rows: list[dict[str, CellValue]],
        out_path: Path,
        sheet_name: str,
    ) -> None:
        """Write rows to a single Excel sheet.

        Args:
            rows: List of row dictionaries to write.
            out_path: Output file path.
            sheet_name: Name for the worksheet.

        Raises:
            WriterError: If writing fails.
        """
        ...

    def write_sheets(
        self,
        sheets: Mapping[str, list[dict[str, CellValue]]],
        out_path: Path,
    ) -> None:
        """Write multiple sheets to an Excel file.

        Args:
            sheets: Mapping of sheet names to row lists.
            out_path: Output file path.

        Raises:
            WriterError: If writing fails.
        """
        ...


__all__ = [
    "DocumentWriterProtocol",
    "ExcelWriterProtocol",
]
