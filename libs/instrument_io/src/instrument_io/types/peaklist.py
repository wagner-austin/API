"""TypedDict definitions for peak list data.

Provides immutable typed structures for chromatogram and mass spectrum
peak lists resulting from integration or peak picking.
"""

from __future__ import annotations

from typing import TypedDict


class ChromatogramPeak(TypedDict):
    """Single peak from chromatogram integration.

    Attributes:
        peak_id: Sequential peak identifier.
        rt_start: Peak start retention time (minutes).
        rt_apex: Peak apex retention time (minutes).
        rt_end: Peak end retention time (minutes).
        area: Integrated peak area.
        height: Peak height at apex.
        width_at_half_height: Peak width at 50% height.
    """

    peak_id: int
    rt_start: float
    rt_apex: float
    rt_end: float
    area: float
    height: float
    width_at_half_height: float


class MassPeak(TypedDict):
    """Single peak from mass spectrum.

    Attributes:
        mz: Mass-to-charge ratio.
        intensity: Absolute intensity.
        relative_intensity: Relative intensity (0-100, base peak = 100).
    """

    mz: float
    intensity: float
    relative_intensity: float


class AnnotatedMassPeak(TypedDict):
    """Mass peak with optional annotation.

    Attributes:
        mz: Mass-to-charge ratio.
        intensity: Absolute intensity.
        relative_intensity: Relative intensity (0-100).
        annotation: Peak annotation (formula, compound name).
        mass_error_ppm: Mass error in ppm (if matched).
    """

    mz: float
    intensity: float
    relative_intensity: float
    annotation: str | None
    mass_error_ppm: float | None


class PeakListMeta(TypedDict):
    """Metadata for a peak list.

    Attributes:
        source_path: Path to source data.
        num_peaks: Number of peaks in list.
        processing_method: Integration/picking method used.
    """

    source_path: str
    num_peaks: int
    processing_method: str


class ChromatogramPeakList(TypedDict):
    """List of chromatogram peaks with metadata.

    Attributes:
        meta: Peak list metadata.
        peaks: List of integrated peaks.
    """

    meta: PeakListMeta
    peaks: list[ChromatogramPeak]


class MassPeakList(TypedDict):
    """List of mass peaks with metadata.

    Attributes:
        meta: Peak list metadata.
        scan_number: Source scan number.
        retention_time: Source retention time.
        peaks: List of mass peaks.
    """

    meta: PeakListMeta
    scan_number: int
    retention_time: float
    peaks: list[MassPeak]


class AnnotatedMassPeakList(TypedDict):
    """List of annotated mass peaks.

    Attributes:
        meta: Peak list metadata.
        scan_number: Source scan number.
        retention_time: Source retention time.
        peaks: List of annotated peaks.
    """

    meta: PeakListMeta
    scan_number: int
    retention_time: float
    peaks: list[AnnotatedMassPeak]


__all__ = [
    "AnnotatedMassPeak",
    "AnnotatedMassPeakList",
    "ChromatogramPeak",
    "ChromatogramPeakList",
    "MassPeak",
    "MassPeakList",
    "PeakListMeta",
]
