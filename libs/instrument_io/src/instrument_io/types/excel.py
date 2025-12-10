"""Excel type definitions for Excel writer.

Provides type aliases for Excel data structures used by ExcelWriter.

All types are immutable with strict typing.
"""

from __future__ import annotations

from collections.abc import Mapping

from instrument_io.types.common import CellValue

# A single row of Excel data: column name -> cell value
ExcelRow = dict[str, CellValue]

# Multiple rows forming a sheet
ExcelRows = list[ExcelRow]

# Multiple sheets: sheet name -> rows
ExcelSheets = Mapping[str, ExcelRows]


__all__ = [
    "ExcelRow",
    "ExcelRows",
    "ExcelSheets",
]
