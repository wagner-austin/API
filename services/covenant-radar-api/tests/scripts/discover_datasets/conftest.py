"""Shared fixtures and protocols for dataset discovery tests."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class CellProtocol(Protocol):
    """Protocol for openpyxl Cell."""

    value: str | int | float | bool | None


class WorksheetProtocol(Protocol):
    """Protocol for openpyxl Worksheet."""

    title: str

    def cell(
        self, row: int, column: int, value: str | int | float | bool | None = None
    ) -> CellProtocol:
        """Get or create cell."""
        ...


class WorkbookProtocol(Protocol):
    """Protocol for openpyxl Workbook."""

    @property
    def active(self) -> WorksheetProtocol:
        """Return active worksheet."""
        ...

    def create_sheet(self, title: str) -> WorksheetProtocol:
        """Create a new worksheet with the given title."""
        ...

    def save(self, filename: str | Path) -> None:
        """Save workbook."""
        ...

    def close(self) -> None:
        """Close workbook."""
        ...

    def remove(self, ws: WorksheetProtocol) -> None:
        """Remove worksheet."""
        ...


class WorkbookCtorProtocol(Protocol):
    """Protocol for Workbook constructor."""

    def __call__(self) -> WorkbookProtocol:
        """Create workbook."""
        ...


def get_workbook_ctor() -> WorkbookCtorProtocol:
    """Get openpyxl Workbook constructor with typing."""
    openpyxl_mod = __import__("openpyxl")
    ctor: WorkbookCtorProtocol = openpyxl_mod.Workbook
    return ctor


class XlwtSheetProtocol(Protocol):
    """Protocol for xlwt Sheet."""

    def write(self, r: int, c: int, label: str | int | float) -> None:
        """Write a value to a cell."""
        ...


class XlwtWorkbookProtocol(Protocol):
    """Protocol for xlwt Workbook."""

    def add_sheet(self, sheetname: str) -> XlwtSheetProtocol:
        """Add a sheet to the workbook."""
        ...

    def save(self, filename: str | Path) -> None:
        """Save the workbook."""
        ...


class XlwtWorkbookCtorProtocol(Protocol):
    """Protocol for xlwt Workbook constructor."""

    def __call__(self) -> XlwtWorkbookProtocol:
        """Create a new workbook."""
        ...


def get_xlwt_workbook_ctor() -> XlwtWorkbookCtorProtocol:
    """Get xlwt Workbook constructor with typing."""
    xlwt_mod = __import__("xlwt")
    ctor: XlwtWorkbookCtorProtocol = xlwt_mod.Workbook
    return ctor
