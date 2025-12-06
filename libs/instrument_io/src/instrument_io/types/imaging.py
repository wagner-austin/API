"""TypedDict definitions for imaging mass spectrometry data.

Provides immutable typed structures for imzML spectra
with spatial coordinate information.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from instrument_io.types.common import MSLevel, Polarity
from instrument_io.types.spectrum import SpectrumData, SpectrumStats


class SpatialCoordinate(TypedDict):
    """Spatial coordinate for imaging MS pixel.

    Attributes:
        x: X coordinate (1-based).
        y: Y coordinate (1-based).
        z: Z coordinate (1-based, None for 2D imaging).
    """

    x: int
    y: int
    z: int | None


class ImagingSpectrumMeta(TypedDict):
    """Metadata for an imaging mass spectrum.

    Attributes:
        source_path: Absolute path to source file.
        index: 0-based index in file.
        coordinate: Spatial (x, y, z) position.
        ms_level: MS level (1 for most imaging MS).
        polarity: Ion polarity mode.
        total_ion_current: Sum of all intensities.
    """

    source_path: str
    index: int
    coordinate: SpatialCoordinate
    ms_level: MSLevel
    polarity: Polarity
    total_ion_current: float


class ImagingSpectrum(TypedDict):
    """Complete imaging mass spectrum with spatial coordinate.

    Attributes:
        meta: Spectrum metadata including coordinate.
        data: Raw m/z and intensity arrays.
        stats: Computed statistics.
    """

    meta: ImagingSpectrumMeta
    data: SpectrumData
    stats: SpectrumStats


# Spectrum mode for imzML data
SpectrumMode = Literal["centroid", "profile", "unknown"]


class ImzMLFileInfo(TypedDict):
    """File-level metadata for imzML dataset.

    Attributes:
        source_path: Path to .imzML file.
        num_spectra: Total number of spectra.
        polarity: Ion polarity (positive, negative, or unknown).
        spectrum_mode: Data acquisition mode.
        x_pixels: Number of pixels in X dimension.
        y_pixels: Number of pixels in Y dimension.
    """

    source_path: str
    num_spectra: int
    polarity: Polarity
    spectrum_mode: SpectrumMode
    x_pixels: int
    y_pixels: int


__all__ = [
    "ImagingSpectrum",
    "ImagingSpectrumMeta",
    "ImzMLFileInfo",
    "SpatialCoordinate",
    "SpectrumMode",
]
