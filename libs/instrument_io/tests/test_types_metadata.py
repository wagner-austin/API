"""Tests for types.metadata module."""

from __future__ import annotations

from instrument_io.types.metadata import (
    AcquisitionInfo,
    BatchInfo,
    FileInfo,
    InstrumentInfo,
    MethodInfo,
    RunInfo,
    SampleInfo,
)


def test_run_info_construction() -> None:
    run = RunInfo(
        path="/data/2024/sample001.D",
        run_id="sample001",
        site="Lab1",
        has_tic=True,
        has_ms=True,
        has_dad=False,
        file_count=5,
    )
    assert run["path"] == "/data/2024/sample001.D"
    assert run["run_id"] == "sample001"
    assert run["site"] == "Lab1"
    assert run["has_tic"] is True
    assert run["has_ms"] is True
    assert run["has_dad"] is False
    assert run["file_count"] == 5


def test_file_info_construction() -> None:
    info = FileInfo(
        path="/data/file.mzML",
        name="file.mzML",
        size_bytes=1024000,
        detector="MS",
        extension=".mzML",
    )
    assert info["path"] == "/data/file.mzML"
    assert info["name"] == "file.mzML"
    assert info["size_bytes"] == 1024000
    assert info["detector"] == "MS"
    assert info["extension"] == ".mzML"


def test_instrument_info_construction() -> None:
    info = InstrumentInfo(
        manufacturer="Agilent",
        model="6545 Q-TOF",
        serial_number="SN123456",
    )
    assert info["manufacturer"] == "Agilent"
    assert info["model"] == "6545 Q-TOF"
    assert info["serial_number"] == "SN123456"


def test_method_info_construction() -> None:
    info = MethodInfo(
        name="Standard_Method.M",
        path="/methods/Standard_Method.M",
        version="1.0",
    )
    assert info["name"] == "Standard_Method.M"
    assert info["path"] == "/methods/Standard_Method.M"
    assert info["version"] == "1.0"


def test_sample_info_construction() -> None:
    info = SampleInfo(
        name="Sample001",
        vial_position="A1",
        injection_volume_ul=1.0,
        dilution_factor=1.0,
    )
    assert info["name"] == "Sample001"
    assert info["vial_position"] == "A1"
    assert info["injection_volume_ul"] == 1.0
    assert info["dilution_factor"] == 1.0


def test_acquisition_info_construction() -> None:
    instrument = InstrumentInfo(
        manufacturer="Agilent",
        model="7890B GC",
        serial_number="US12345678",
    )
    method = MethodInfo(
        name="GC_Method.M",
        path="/methods/GC_Method.M",
        version="2.1",
    )
    sample = SampleInfo(
        name="Sample001",
        vial_position="A1",
        injection_volume_ul=1.0,
        dilution_factor=1.0,
    )
    info = AcquisitionInfo(
        instrument=instrument,
        method=method,
        sample=sample,
        acquisition_date="2024-01-15T10:00:00",
        operator="JSmith",
    )
    assert info["instrument"]["manufacturer"] == "Agilent"
    assert info["method"]["name"] == "GC_Method.M"
    assert info["sample"]["name"] == "Sample001"
    assert info["acquisition_date"] == "2024-01-15T10:00:00"
    assert info["operator"] == "JSmith"


def test_batch_info_construction() -> None:
    run1 = RunInfo(
        path="/data/batch001/run001.D",
        run_id="run001",
        site="Lab1",
        has_tic=True,
        has_ms=True,
        has_dad=False,
        file_count=3,
    )
    run2 = RunInfo(
        path="/data/batch001/run002.D",
        run_id="run002",
        site="Lab1",
        has_tic=True,
        has_ms=True,
        has_dad=True,
        file_count=4,
    )
    info = BatchInfo(
        path="/data/batch001",
        batch_id="batch001",
        run_count=2,
        runs=[run1, run2],
    )
    assert info["path"] == "/data/batch001"
    assert info["batch_id"] == "batch001"
    assert info["run_count"] == 2
    assert len(info["runs"]) == 2
    assert info["runs"][0]["run_id"] == "run001"
    assert info["runs"][1]["run_id"] == "run002"


def test_batch_info_empty_runs() -> None:
    info = BatchInfo(
        path="/data/empty_batch",
        batch_id="empty_batch",
        run_count=0,
        runs=[],
    )
    assert info["run_count"] == 0
    assert info["runs"] == []
