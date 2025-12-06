"""Tests for _exceptions module."""

from __future__ import annotations

from instrument_io._exceptions import (
    AgilentReadError,
    CSVReadError,
    DecodingError,
    ExcelReadError,
    ImzMLReadError,
    InstrumentIOError,
    MGFReadError,
    MzMLReadError,
    ThermoReadError,
    UnsupportedFormatError,
    WatersReadError,
    WriterError,
)


def test_instrument_io_error_message() -> None:
    err = InstrumentIOError("test error")
    assert str(err) == "test error"


def test_unsupported_format_error() -> None:
    err = UnsupportedFormatError("/path/to/file", "Unknown format")
    assert "/path/to/file" in str(err)
    assert "Unknown format" in str(err)
    assert err.path == "/path/to/file"
    assert err.message == "Unknown format"


def test_agilent_read_error() -> None:
    err = AgilentReadError("/data/sample.D", "No TIC data")
    assert "/data/sample.D" in str(err)
    assert "No TIC data" in str(err)
    assert err.path == "/data/sample.D"
    assert err.message == "No TIC data"


def test_mzml_read_error() -> None:
    err = MzMLReadError("/data/sample.mzML", "Invalid spectrum")
    assert "/data/sample.mzML" in str(err)
    assert "Invalid spectrum" in str(err)
    assert err.path == "/data/sample.mzML"
    assert err.message == "Invalid spectrum"


def test_excel_read_error() -> None:
    err = ExcelReadError("/data/sample.xlsx", "Sheet not found")
    assert "/data/sample.xlsx" in str(err)
    assert "Sheet not found" in str(err)
    assert err.path == "/data/sample.xlsx"
    assert err.message == "Sheet not found"


def test_decoding_error() -> None:
    err = DecodingError("ms_level", "Invalid value: 5")
    assert "ms_level" in str(err)
    assert "Invalid value: 5" in str(err)
    assert err.context == "ms_level"
    assert err.message == "Invalid value: 5"


def test_writer_error() -> None:
    err = WriterError("/output/file.xlsx", "Cannot create file")
    assert "/output/file.xlsx" in str(err)
    assert "Cannot create file" in str(err)
    assert err.path == "/output/file.xlsx"
    assert err.message == "Cannot create file"


def test_csv_read_error() -> None:
    err = CSVReadError("/data/chromatogram.csv", "Column not found")
    assert "/data/chromatogram.csv" in str(err)
    assert "Column not found" in str(err)
    assert err.path == "/data/chromatogram.csv"
    assert err.message == "Column not found"


def test_thermo_read_error() -> None:
    err = ThermoReadError("/data/sample.raw", "Cannot open file")
    assert "/data/sample.raw" in str(err)
    assert "Cannot open file" in str(err)
    assert err.path == "/data/sample.raw"
    assert err.message == "Cannot open file"


def test_mgf_read_error() -> None:
    err = MGFReadError("/data/sample.mgf", "Missing pepmass")
    assert "/data/sample.mgf" in str(err)
    assert "Missing pepmass" in str(err)
    assert err.path == "/data/sample.mgf"
    assert err.message == "Missing pepmass"


def test_imzml_read_error() -> None:
    err = ImzMLReadError("/data/sample.imzML", "Missing .ibd file")
    assert "/data/sample.imzML" in str(err)
    assert "Missing .ibd file" in str(err)
    assert err.path == "/data/sample.imzML"
    assert err.message == "Missing .ibd file"


def test_waters_read_error() -> None:
    err = WatersReadError("/data/sample.raw", "No MS data found")
    assert "/data/sample.raw" in str(err)
    assert "No MS data found" in str(err)
    assert err.path == "/data/sample.raw"
    assert err.message == "No MS data found"


def test_exception_inheritance() -> None:
    assert issubclass(UnsupportedFormatError, InstrumentIOError)
    assert issubclass(AgilentReadError, InstrumentIOError)
    assert issubclass(MzMLReadError, InstrumentIOError)
    assert issubclass(ExcelReadError, InstrumentIOError)
    assert issubclass(DecodingError, InstrumentIOError)
    assert issubclass(WriterError, InstrumentIOError)
    assert issubclass(CSVReadError, InstrumentIOError)
    assert issubclass(ThermoReadError, InstrumentIOError)
    assert issubclass(MGFReadError, InstrumentIOError)
    assert issubclass(ImzMLReadError, InstrumentIOError)
    assert issubclass(WatersReadError, InstrumentIOError)
