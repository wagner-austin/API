"""Tests for _protocols.pdfplumber module."""

from __future__ import annotations

from pathlib import Path

import pytest

from instrument_io._protocols.pdfplumber import _open_pdf


def test_open_pdf_with_nonexistent_file(tmp_path: Path) -> None:
    """Test that _open_pdf raises when file doesn't exist."""
    nonexistent = tmp_path / "nonexistent.pdf"
    with pytest.raises(FileNotFoundError):
        _open_pdf(nonexistent)
