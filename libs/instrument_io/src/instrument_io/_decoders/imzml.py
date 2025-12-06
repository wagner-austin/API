"""Decoder functions for imzML imaging mass spectrometry data.

Converts pyimzML data structures to typed structures.
"""

from __future__ import annotations

from instrument_io.types.common import MSLevel, Polarity
from instrument_io.types.imaging import (
    ImagingSpectrumMeta,
    SpatialCoordinate,
    SpectrumMode,
)
from instrument_io.types.spectrum import SpectrumData, SpectrumStats


def _decode_coordinate(coord: tuple[int, int, int]) -> SpatialCoordinate:
    """Decode coordinate tuple to SpatialCoordinate TypedDict.

    Args:
        coord: (x, y, z) coordinate tuple from pyimzML.

    Returns:
        SpatialCoordinate TypedDict.
    """
    x, y, z = coord
    # z=1 is the default for 2D imaging, treat as None
    z_value: int | None = None if z == 1 else z
    return SpatialCoordinate(x=x, y=y, z=z_value)


def _decode_imzml_polarity(polarity_str: str) -> Polarity:
    """Decode polarity string from pyimzML to Polarity literal.

    Args:
        polarity_str: Polarity from parser ('positive', 'negative', 'mixed').

    Returns:
        Polarity literal.
    """
    polarity_lower = polarity_str.lower()
    if polarity_lower == "positive":
        return "positive"
    if polarity_lower == "negative":
        return "negative"
    return "unknown"


def _decode_spectrum_mode(mode_str: str) -> SpectrumMode:
    """Decode spectrum mode string to SpectrumMode literal.

    Args:
        mode_str: Spectrum mode from parser ('centroid', 'profile').

    Returns:
        SpectrumMode literal.
    """
    mode_lower = mode_str.lower()
    if mode_lower == "centroid":
        return "centroid"
    if mode_lower == "profile":
        return "profile"
    return "unknown"


def _compute_imzml_spectrum_stats(
    mz_values: list[float],
    intensities: list[float],
) -> SpectrumStats:
    """Compute statistics from spectrum data.

    Args:
        mz_values: List of m/z values.
        intensities: List of intensity values.

    Returns:
        SpectrumStats TypedDict.
    """
    if not mz_values or not intensities:
        return SpectrumStats(
            num_peaks=0,
            mz_min=0.0,
            mz_max=0.0,
            base_peak_mz=0.0,
            base_peak_intensity=0.0,
        )

    num_peaks = len(mz_values)
    mz_min = min(mz_values)
    mz_max = max(mz_values)

    max_intensity = max(intensities)
    max_idx = intensities.index(max_intensity)
    base_peak_mz = mz_values[max_idx]

    return SpectrumStats(
        num_peaks=num_peaks,
        mz_min=mz_min,
        mz_max=mz_max,
        base_peak_mz=base_peak_mz,
        base_peak_intensity=max_intensity,
    )


def _make_imzml_spectrum_data(
    mz_values: list[float],
    intensities: list[float],
) -> SpectrumData:
    """Create SpectrumData TypedDict.

    Args:
        mz_values: List of m/z values.
        intensities: List of intensity values.

    Returns:
        SpectrumData TypedDict.
    """
    return SpectrumData(
        mz_values=mz_values,
        intensities=intensities,
    )


def _make_imzml_spectrum_meta(
    source_path: str,
    index: int,
    coordinate: SpatialCoordinate,
    polarity: Polarity,
    total_ion_current: float,
) -> ImagingSpectrumMeta:
    """Create ImagingSpectrumMeta TypedDict.

    Args:
        source_path: Path to source file.
        index: 0-based index in file.
        coordinate: Spatial coordinate.
        polarity: Ion polarity.
        total_ion_current: TIC value.

    Returns:
        ImagingSpectrumMeta TypedDict.
    """
    ms_level: MSLevel = 1  # Imaging MS is typically MS1
    return ImagingSpectrumMeta(
        source_path=source_path,
        index=index,
        coordinate=coordinate,
        ms_level=ms_level,
        polarity=polarity,
        total_ion_current=total_ion_current,
    )


__all__ = [
    "_compute_imzml_spectrum_stats",
    "_decode_coordinate",
    "_decode_imzml_polarity",
    "_decode_spectrum_mode",
    "_make_imzml_spectrum_data",
    "_make_imzml_spectrum_meta",
]
