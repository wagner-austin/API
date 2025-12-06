"""Integration tests for SMPS reader using actual files."""

from __future__ import annotations

from io import TextIOWrapper
from pathlib import Path
from typing import Protocol

import pytest

from instrument_io._exceptions import SMPSReadError
from instrument_io.readers.smps import SMPSReader


class _PathOpenMethod(Protocol):
    """Protocol for Path.open method (unbound)."""

    def __call__(
        self,
        path_self: Path,
        mode: str = ...,
        buffering: int = ...,
        encoding: str | None = ...,
        errors: str | None = ...,
        newline: str | None = ...,
    ) -> TextIOWrapper:
        """Open a file at path."""
        ...


# Method name constant to prevent ruff from simplifying getattr
_OPEN_METHOD = "open"

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_RPS = FIXTURES_DIR / "sample.rps"
COMPLEX_RPS = FIXTURES_DIR / "complex.rps"
CP1252_RPS = FIXTURES_DIR / "cp1252.rps"
CP1252_REAL_RPS = FIXTURES_DIR / "cp1252_real.rps"
EMPTY_PARAMS_RPS = FIXTURES_DIR / "empty_params.rps"
ALL_EMPTY_HEADERS_RPS = FIXTURES_DIR / "all_empty_headers.rps"


def test_read_metadata() -> None:
    """Test reading metadata from actual SMPS file."""
    reader = SMPSReader()
    metadata = reader.read_metadata(SAMPLE_RPS)

    assert metadata["timestamp"] == "10/30/2025 11:20 AM"
    assert metadata["instrument"] == "nsmps"
    assert metadata["lower_voltage_limit"] == 10.0
    assert metadata["upper_voltage_limit"] == 10000.0
    assert metadata["sample_duration"] == 30.5


def test_read_data() -> None:
    """Test reading data rows from actual SMPS file."""
    reader = SMPSReader()
    data = reader.read_data(SAMPLE_RPS)

    assert len(data) == 3
    assert data[0]["Sample Duration [s]"] == 3.01
    assert data[0]["Bin End Voltage [V]"] == 12.62
    assert data[0]["Count [#]"] == 100

    assert data[1]["Sample Duration [s]"] == 3.02
    assert data[1]["Count [#]"] == 150

    assert data[2]["Sample Duration [s]"] == 3.03
    assert data[2]["Count [#]"] == 120


def test_read_full() -> None:
    """Test reading complete SMPS file."""
    reader = SMPSReader()
    result = reader.read_full(SAMPLE_RPS)

    # Check metadata
    assert result["metadata"]["timestamp"] == "10/30/2025 11:20 AM"
    assert result["metadata"]["instrument"] == "nsmps"

    # Check data
    assert len(result["data"]) == 3
    assert result["data"][0]["Count [#]"] == 100


def test_supports_format_rps() -> None:
    """Test format detection for .rps files."""
    reader = SMPSReader()
    assert reader.supports_format(SAMPLE_RPS) is True


def test_supports_format_non_rps() -> None:
    """Test format detection for non-.rps files."""
    reader = SMPSReader()
    non_rps = FIXTURES_DIR / "sample.txt"
    assert reader.supports_format(non_rps) is False


def test_read_metadata_file_not_exists() -> None:
    """Test error when file doesn't exist."""
    reader = SMPSReader()
    nonexistent = FIXTURES_DIR / "nonexistent.rps"

    with pytest.raises(SMPSReadError) as exc_info:
        reader.read_metadata(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_metadata_not_rps_file() -> None:
    """Test error when file is not an .rps file."""
    reader = SMPSReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(SMPSReadError) as exc_info:
        reader.read_metadata(txt_file)

    assert "Not an SMPS .rps file" in str(exc_info.value)


def test_read_data_file_not_exists() -> None:
    """Test error when file doesn't exist for read_data."""
    reader = SMPSReader()
    nonexistent = FIXTURES_DIR / "nonexistent.rps"

    with pytest.raises(SMPSReadError) as exc_info:
        reader.read_data(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_data_not_rps_file() -> None:
    """Test error when file is not an .rps file for read_data."""
    reader = SMPSReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(SMPSReadError) as exc_info:
        reader.read_data(txt_file)

    assert "Not an SMPS .rps file" in str(exc_info.value)


def test_read_full_file_not_exists() -> None:
    """Test error when file doesn't exist for read_full."""
    reader = SMPSReader()
    nonexistent = FIXTURES_DIR / "nonexistent.rps"

    with pytest.raises(SMPSReadError) as exc_info:
        reader.read_full(nonexistent)

    assert "File does not exist" in str(exc_info.value)


def test_read_full_not_rps_file() -> None:
    """Test error when file is not an .rps file for read_full."""
    reader = SMPSReader()
    txt_file = FIXTURES_DIR / "sample.txt"

    with pytest.raises(SMPSReadError) as exc_info:
        reader.read_full(txt_file)

    assert "Not an SMPS .rps file" in str(exc_info.value)


def test_read_complex_rps() -> None:
    """Test reading complex real RPS file from Faiola Lab."""
    reader = SMPSReader()

    # Test reading metadata from complex file
    metadata = reader.read_metadata(COMPLEX_RPS)
    assert "timestamp" in metadata
    assert "instrument" in metadata
    assert "lower_voltage_limit" in metadata
    assert metadata["lower_voltage_limit"] > 0

    # Test reading data from complex file
    data = reader.read_data(COMPLEX_RPS)
    assert len(data) > 10  # Should have multiple data rows

    # Test reading full file
    result = reader.read_full(COMPLEX_RPS)
    assert result["metadata"] == metadata
    assert result["data"] == data


def test_read_cp1252_encoded_rps() -> None:
    """Test reading CP1252 encoded RPS file (triggers encoding fallback)."""
    reader = SMPSReader()

    # This file is CP1252 encoded and will fail UTF-8, triggering fallback
    metadata = reader.read_metadata(CP1252_RPS)
    assert metadata["timestamp"] == "10/30/2025 11:20 AM"
    assert metadata["instrument"] == "nsmps"
    assert metadata["lower_voltage_limit"] == 10.0

    # Test reading data with CP1252 encoding
    data = reader.read_data(CP1252_RPS)
    assert len(data) == 1
    assert data[0]["Count [#]"] == 100


def test_read_cp1252_real_encoding_fallback() -> None:
    """Test reading file with actual CP1252 byte (0x80 = €) that fails UTF-8.

    This file contains byte 0x80 which is invalid UTF-8 but valid CP1252 (€).
    This triggers the encoding fallback from UTF-8 to CP1252.
    """
    reader = SMPSReader()

    # This file has 0x80 byte which is invalid UTF-8, triggers CP1252 fallback
    metadata = reader.read_metadata(CP1252_REAL_RPS)
    assert metadata["timestamp"] == "10/30/2025 11:20 AM"
    # Instrument name contains € character from CP1252 encoding
    assert "nsmps" in metadata["instrument"]
    assert metadata["lower_voltage_limit"] == 10.0

    # Test reading data
    data = reader.read_data(CP1252_REAL_RPS)
    assert len(data) == 1
    assert data[0]["Count [#]"] == 100


def test_read_empty_params_rps() -> None:
    """Test reading SMPS file with empty parameter names.

    This covers the decoder branch where some param names are empty and skipped.
    """
    reader = SMPSReader()

    # Read metadata - empty param names should be skipped
    metadata = reader.read_metadata(EMPTY_PARAMS_RPS)
    assert metadata["timestamp"] == "10/30/2025 11:20 AM"
    assert metadata["instrument"] == "nsmps"

    # Read data - empty column headers should be skipped
    data = reader.read_data(EMPTY_PARAMS_RPS)
    assert len(data) == 1

    # Only non-empty headers should be in the result
    assert "Col2" in data[0]
    assert "Col4" in data[0]


def test_read_all_empty_headers_rps() -> None:
    """Test reading SMPS file where ALL column headers are empty.

    This covers the decoder branch where row_dict is empty after processing.
    """
    reader = SMPSReader()

    # Read data - all column headers empty, rows should be skipped
    data = reader.read_data(ALL_EMPTY_HEADERS_RPS)

    # Table should be empty (all headers empty, so no data extracted)
    assert len(data) == 0


def test_read_invalid_both_encodings_rps() -> None:
    """Test reading file that fails both UTF-8 and CP1252 encoding.

    This covers smps.py lines 46-47 where cp1252 fallback also fails.
    The file contains byte 0x81 which is invalid in both encodings.
    """
    reader = SMPSReader()
    invalid_file = FIXTURES_DIR / "invalid_both_encodings.rps"

    with pytest.raises(SMPSReadError) as exc_info:
        reader.read_metadata(invalid_file)

    assert "Failed to read file" in str(exc_info.value)


def test_read_file_oserror_on_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OSError during initial file open.

    This covers smps.py lines 48-49 where OSError occurs on initial open.
    We monkeypatch Path.open to raise OSError on the first call.
    """
    from instrument_io.readers import smps

    # Create a valid .rps file
    valid_rps = tmp_path / "test.rps"
    valid_rps.write_text("test content", encoding="utf-8")

    # Use getattr with Protocol type annotation to get unbound Path.open
    original_path_open: _PathOpenMethod = getattr(Path, _OPEN_METHOD)

    def failing_open(
        path_self: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> TextIOWrapper:
        if path_self.suffix == ".rps":
            raise OSError("Simulated disk error")
        return original_path_open(path_self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, _OPEN_METHOD, failing_open)

    with pytest.raises(SMPSReadError) as exc_info:
        smps._read_lines(valid_rps)

    assert "Failed to read file" in str(exc_info.value)
