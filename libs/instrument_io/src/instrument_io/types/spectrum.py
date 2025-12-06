"""TypedDict definitions for mass spectrum data.

Provides immutable typed structures for MS and MS/MS spectra
with associated metadata and statistics.
"""

from __future__ import annotations

from typing import TypedDict

from instrument_io.types.common import MSLevel, Polarity


class SpectrumMeta(TypedDict):
    """Metadata for a mass spectrum.

    Attributes:
        source_path: Absolute path to source file.
        scan_number: Scan index (1-based).
        retention_time: Retention time in minutes.
        ms_level: MS level (1, 2, or 3).
        polarity: Ion polarity mode.
        total_ion_current: Sum of all intensities.
    """

    source_path: str
    scan_number: int
    retention_time: float
    ms_level: MSLevel
    polarity: Polarity
    total_ion_current: float


class SpectrumData(TypedDict):
    """Raw mass spectrum data.

    Attributes:
        mz_values: Mass-to-charge ratios.
        intensities: Ion intensities.
    """

    mz_values: list[float]
    intensities: list[float]


class SpectrumStats(TypedDict):
    """Computed statistics for a mass spectrum.

    Attributes:
        num_peaks: Number of peaks/data points.
        mz_min: Minimum m/z value.
        mz_max: Maximum m/z value.
        base_peak_mz: m/z of most intense peak.
        base_peak_intensity: Intensity of most intense peak.
    """

    num_peaks: int
    mz_min: float
    mz_max: float
    base_peak_mz: float
    base_peak_intensity: float


class MSSpectrum(TypedDict):
    """Complete mass spectrum with metadata and data.

    Attributes:
        meta: Spectrum metadata.
        data: Raw m/z and intensity arrays.
        stats: Computed statistics.
    """

    meta: SpectrumMeta
    data: SpectrumData
    stats: SpectrumStats


class PrecursorInfo(TypedDict):
    """Precursor ion information for MS/MS spectra.

    Attributes:
        mz: Precursor m/z value.
        charge: Charge state (None if unknown).
        intensity: Precursor intensity (None if unknown).
        isolation_window: Isolation window width (None if unknown).
    """

    mz: float
    charge: int | None
    intensity: float | None
    isolation_window: float | None


class MS2Spectrum(TypedDict):
    """MS/MS spectrum with precursor information.

    Attributes:
        meta: Spectrum metadata (ms_level should be 2).
        precursor: Precursor ion information.
        data: Raw m/z and intensity arrays.
        stats: Computed statistics.
    """

    meta: SpectrumMeta
    precursor: PrecursorInfo
    data: SpectrumData
    stats: SpectrumStats


class MS3Spectrum(TypedDict):
    """MS3 spectrum with precursor chain.

    Attributes:
        meta: Spectrum metadata (ms_level should be 3).
        precursors: List of precursor ions (MS1 -> MS2).
        data: Raw m/z and intensity arrays.
        stats: Computed statistics.
    """

    meta: SpectrumMeta
    precursors: list[PrecursorInfo]
    data: SpectrumData
    stats: SpectrumStats


__all__ = [
    "MS2Spectrum",
    "MS3Spectrum",
    "MSSpectrum",
    "PrecursorInfo",
    "SpectrumData",
    "SpectrumMeta",
    "SpectrumStats",
]
