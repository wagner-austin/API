"""TypedDict definitions for chromatogram data.

Provides immutable typed structures for TIC, EIC, and DAD chromatograms
with associated metadata and statistics.
"""

from __future__ import annotations

from typing import TypedDict

from instrument_io.types.common import SignalType


class ChromatogramMeta(TypedDict):
    """Metadata extracted from chromatogram source.

    Attributes:
        source_path: Absolute path to source file/directory.
        instrument: Instrument name or model.
        method_name: Acquisition method name.
        sample_name: Sample identifier.
        acquisition_date: ISO format date string.
        signal_type: Type of signal (TIC, EIC, DAD, etc.).
        detector: Detector identifier string.
    """

    source_path: str
    instrument: str
    method_name: str
    sample_name: str
    acquisition_date: str
    signal_type: SignalType
    detector: str


class ChromatogramData(TypedDict):
    """Raw chromatogram data points.

    Attributes:
        retention_times: Time points in minutes.
        intensities: Signal intensity at each time point.
    """

    retention_times: list[float]
    intensities: list[float]


class ChromatogramStats(TypedDict):
    """Computed statistics for a chromatogram.

    Attributes:
        num_points: Number of data points.
        rt_min: Minimum retention time (minutes).
        rt_max: Maximum retention time (minutes).
        rt_step_mean: Average time between points.
        intensity_min: Minimum intensity value.
        intensity_max: Maximum intensity value.
        intensity_mean: Mean intensity.
        intensity_p99: 99th percentile intensity.
    """

    num_points: int
    rt_min: float
    rt_max: float
    rt_step_mean: float
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    intensity_p99: float


class EICParams(TypedDict):
    """Parameters for EIC extraction.

    Attributes:
        target_mz: Target m/z value for extraction.
        mz_tolerance: Tolerance window (Daltons).
    """

    target_mz: float
    mz_tolerance: float


class TICData(TypedDict):
    """Total Ion Chromatogram data.

    Complete TIC with metadata, raw data, and statistics.

    Attributes:
        meta: Chromatogram metadata.
        data: Raw time/intensity data.
        stats: Computed statistics.
    """

    meta: ChromatogramMeta
    data: ChromatogramData
    stats: ChromatogramStats


class EICData(TypedDict):
    """Extracted Ion Chromatogram data.

    Complete EIC with extraction parameters.

    Attributes:
        meta: Chromatogram metadata.
        params: EIC extraction parameters.
        data: Raw time/intensity data.
        stats: Computed statistics.
    """

    meta: ChromatogramMeta
    params: EICParams
    data: ChromatogramData
    stats: ChromatogramStats


class DADSlice(TypedDict):
    """Single wavelength slice from DAD data.

    Represents chromatogram at a specific wavelength.

    Attributes:
        meta: Chromatogram metadata.
        wavelength_nm: Wavelength in nanometers.
        data: Raw time/intensity data.
        stats: Computed statistics.
    """

    meta: ChromatogramMeta
    wavelength_nm: float
    data: ChromatogramData
    stats: ChromatogramStats


class DADData(TypedDict):
    """Full DAD (Diode Array Detector) data.

    Contains multiple wavelength slices.

    Attributes:
        meta: Chromatogram metadata.
        wavelengths: List of wavelengths in nm.
        retention_times: Shared time axis.
        intensity_matrix: 2D matrix [wavelength_idx][time_idx].
    """

    meta: ChromatogramMeta
    wavelengths: list[float]
    retention_times: list[float]
    intensity_matrix: list[list[float]]


__all__ = [
    "ChromatogramData",
    "ChromatogramMeta",
    "ChromatogramStats",
    "DADData",
    "DADSlice",
    "EICData",
    "EICParams",
    "TICData",
]
