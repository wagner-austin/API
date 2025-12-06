"""TypedDict definitions for instrument and run metadata.

Provides immutable typed structures for instrument configuration,
acquisition methods, sample information, and run directories.
"""

from __future__ import annotations

from typing import TypedDict


class InstrumentInfo(TypedDict):
    """Instrument identification.

    Attributes:
        manufacturer: Instrument manufacturer (e.g., "Agilent").
        model: Instrument model (e.g., "7890B GC").
        serial_number: Instrument serial number.
    """

    manufacturer: str
    model: str
    serial_number: str


class MethodInfo(TypedDict):
    """Acquisition method information.

    Attributes:
        name: Method name.
        path: Full path to method file.
        version: Method version string.
    """

    name: str
    path: str
    version: str


class SampleInfo(TypedDict):
    """Sample information from sequence.

    Attributes:
        name: Sample name/identifier.
        vial_position: Autosampler vial position.
        injection_volume_ul: Injection volume in microliters.
        dilution_factor: Sample dilution factor.
    """

    name: str
    vial_position: str
    injection_volume_ul: float
    dilution_factor: float


class AcquisitionInfo(TypedDict):
    """Complete acquisition metadata.

    Attributes:
        instrument: Instrument identification.
        method: Acquisition method.
        sample: Sample information.
        acquisition_date: ISO format acquisition date.
        operator: Operator name.
    """

    instrument: InstrumentInfo
    method: MethodInfo
    sample: SampleInfo
    acquisition_date: str
    operator: str


class RunInfo(TypedDict):
    """Information about a single GC/LC run directory.

    Attributes:
        path: Absolute path to run directory.
        run_id: Run identifier extracted from path.
        site: Site/location identifier.
        has_tic: Whether TIC data is available.
        has_ms: Whether MS data is available.
        has_dad: Whether DAD data is available.
        file_count: Number of data files in run.
    """

    path: str
    run_id: str
    site: str
    has_tic: bool
    has_ms: bool
    has_dad: bool
    file_count: int


class BatchInfo(TypedDict):
    """Information about a batch of runs.

    Attributes:
        path: Absolute path to batch directory.
        batch_id: Batch identifier.
        run_count: Number of runs in batch.
        runs: List of run info for each run.
    """

    path: str
    batch_id: str
    run_count: int
    runs: list[RunInfo]


class FileInfo(TypedDict):
    """Information about a single data file.

    Attributes:
        path: Absolute path to file.
        name: Filename.
        size_bytes: File size in bytes.
        detector: Detector type.
        extension: File extension.
    """

    path: str
    name: str
    size_bytes: int
    detector: str
    extension: str


__all__ = [
    "AcquisitionInfo",
    "BatchInfo",
    "FileInfo",
    "InstrumentInfo",
    "MethodInfo",
    "RunInfo",
    "SampleInfo",
]
