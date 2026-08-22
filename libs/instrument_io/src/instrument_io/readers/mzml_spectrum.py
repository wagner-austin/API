"""Spectrum decoding helpers for the mzML/mzXML reader.

The reader class lives in :mod:`instrument_io.readers.mzml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from instrument_io._decoders.mzml import (
    _compute_spectrum_stats,
    _decode_intensity_array,
    _decode_ms_level,
    _decode_mz_array,
    _decode_polarity,
    _decode_retention_time,
    _decode_scan_number,
    _make_spectrum_data,
)
from instrument_io._exceptions import MzMLReadError
from instrument_io._protocols.pyteomics import (
    SpectrumDictProtocol,
)
from instrument_io.types.chromatogram import (
    ChromatogramStats,
)
from instrument_io.types.spectrum import (
    MS2Spectrum,
    MSSpectrum,
    PrecursorInfo,
    SpectrumMeta,
)


@runtime_checkable
class _ArrayLikeProtocol(Protocol):
    """Protocol for objects with tolist() method."""

    def tolist(self) -> list[float]: ...


def _compute_chromatogram_stats(
    retention_times: list[float],
    intensities: list[float],
) -> ChromatogramStats:
    """Compute statistics for chromatogram data.

    Args:
        retention_times: List of retention times.
        intensities: List of intensities.

    Returns:
        ChromatogramStats TypedDict.
    """
    n = len(retention_times)

    if n == 0:
        return ChromatogramStats(
            num_points=0,
            rt_min=0.0,
            rt_max=0.0,
            rt_step_mean=0.0,
            intensity_min=0.0,
            intensity_max=0.0,
            intensity_mean=0.0,
            intensity_p99=0.0,
        )

    rt_min = min(retention_times)
    rt_max = max(retention_times)
    rt_step_mean = (rt_max - rt_min) / (n - 1) if n > 1 else 0.0

    intensity_min = min(intensities)
    intensity_max = max(intensities)
    intensity_mean = sum(intensities) / n

    # Compute 99th percentile (int(n * 0.99) is always < n for n > 0)
    sorted_intensities = sorted(intensities)
    intensity_p99 = sorted_intensities[int(n * 0.99)]

    return ChromatogramStats(
        num_points=n,
        rt_min=rt_min,
        rt_max=rt_max,
        rt_step_mean=rt_step_mean,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        intensity_mean=intensity_mean,
        intensity_p99=intensity_p99,
    )


def _is_mzml_file(path: Path) -> bool:
    """Check if path is an mzML file."""
    return path.is_file() and path.suffix.lower() == ".mzml"


def _is_mzxml_file(path: Path) -> bool:
    """Check if path is an mzXML file."""
    return path.is_file() and path.suffix.lower() == ".mzxml"


def _extract_array_from_spectrum(
    spectrum: SpectrumDictProtocol,
    key: str,
    source_path: str,
) -> list[float]:
    """Extract array from spectrum dict and convert to list.

    Args:
        spectrum: Spectrum dictionary from pyteomics.
        key: Array key (e.g., "m/z array", "intensity array").

    Returns:
        List of float values.

    Raises:
        MzMLReadError: If array not found or invalid type.
    """
    if key not in spectrum:
        raise MzMLReadError(source_path, f"Missing required array: {key}")

    value = spectrum[key]
    # Check if value has tolist() method (numpy array or similar)
    if not isinstance(value, _ArrayLikeProtocol):
        raise MzMLReadError(
            source_path,
            f"Expected array for key '{key}', got {type(value).__name__}",
        )

    arr_list: list[float] = value.tolist()
    return arr_list


def _extract_float_or_zero(
    spectrum: SpectrumDictProtocol,
    key: str,
) -> float:
    """Extract float value from spectrum, defaulting to 0.0.

    Args:
        spectrum: Spectrum dictionary.
        key: Key to extract.

    Returns:
        Float value or 0.0 if not found.
    """
    value = spectrum.get(key)
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _extract_polarity_string(spectrum: SpectrumDictProtocol) -> str | None:
    """Extract polarity string from spectrum metadata.

    Args:
        spectrum: Spectrum dictionary.

    Returns:
        Polarity string or None if not found.
    """
    # pyteomics uses various keys for polarity
    for key in ["positive scan", "negative scan", "polarity"]:
        if key in spectrum:
            value = spectrum.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, bool) and value:
                if "positive" in key:
                    return "positive"
                if "negative" in key:
                    return "negative"
    return None


def _extract_precursor_info(spectrum: SpectrumDictProtocol) -> PrecursorInfo | None:
    """Extract precursor information from MS2 spectrum.

    Args:
        spectrum: Spectrum dictionary.

    Returns:
        PrecursorInfo or None if not MS2.
    """
    if "precursorMz" not in spectrum and "precursor" not in spectrum:
        return None

    # Try direct precursorMz (mzXML style)
    precursor_mz = spectrum.get("precursorMz")
    if precursor_mz is not None:
        mz_val: float
        if isinstance(precursor_mz, list) and len(precursor_mz) > 0:
            first_item = precursor_mz[0]
            mz_val = float(first_item) if isinstance(first_item, (int, float)) else 0.0
        elif isinstance(precursor_mz, (int, float)):
            mz_val = float(precursor_mz)
        else:
            mz_val = 0.0

        return PrecursorInfo(
            mz=mz_val,
            charge=None,
            intensity=None,
            isolation_window=None,
        )

    # Try precursor list (mzML style)
    precursor_list = spectrum.get("precursor")
    if isinstance(precursor_list, list) and len(precursor_list) > 0:
        first_precursor = precursor_list[0]
        if isinstance(first_precursor, dict):
            selected_ions = first_precursor.get("selectedIons", [])
            if isinstance(selected_ions, list) and len(selected_ions) > 0:
                ion = selected_ions[0]
                if isinstance(ion, dict):
                    mz_value = ion.get("selected ion m/z", 0.0)
                    charge_value = ion.get("charge state")
                    intensity_value = ion.get("peak intensity")

                    mz_float = float(mz_value) if isinstance(mz_value, (int, float)) else 0.0
                    charge_int: int | None = (
                        int(charge_value) if isinstance(charge_value, (int, float)) else None
                    )
                    intensity_float: float | None = (
                        float(intensity_value)
                        if isinstance(intensity_value, (int, float))
                        else None
                    )

                    return PrecursorInfo(
                        mz=mz_float,
                        charge=charge_int,
                        intensity=intensity_float,
                        isolation_window=None,
                    )

    return None


def _spectrum_to_msspectrum(
    spectrum: SpectrumDictProtocol,
    source_path: str,
) -> MSSpectrum:
    """Convert pyteomics spectrum dict to MSSpectrum TypedDict.

    Args:
        spectrum: Spectrum dictionary from pyteomics.
        source_path: Path to source file.

    Returns:
        MSSpectrum TypedDict.
    """
    # Extract arrays
    mz_raw = _extract_array_from_spectrum(spectrum, "m/z array", source_path)
    intensity_raw = _extract_array_from_spectrum(spectrum, "intensity array", source_path)

    mz_values = _decode_mz_array(mz_raw)
    intensities = _decode_intensity_array(intensity_raw)

    # Extract metadata
    scan_id = spectrum.get("id")
    scan_str: str | int | None = scan_id if isinstance(scan_id, (str, int)) else None
    scan_number = _decode_scan_number(scan_str)

    rt_raw = spectrum.get("scanList")
    rt_value: float = 0.0
    if isinstance(rt_raw, dict):
        scans = rt_raw.get("scan")
        if isinstance(scans, list) and len(scans) > 0:
            first_scan = scans[0]
            if isinstance(first_scan, dict):
                # Try "scan start time" key
                scan_time = first_scan.get("scan start time")
                if isinstance(scan_time, (int, float)):
                    rt_value = float(scan_time)
    else:
        # Try direct retention time
        direct_rt = spectrum.get("retentionTime")
        if isinstance(direct_rt, (int, float)):
            rt_value = float(direct_rt)

    retention_time = _decode_retention_time(rt_value)

    # MS level
    ms_level_raw_ml = spectrum.get("ms level")
    ms_level_val_raw: int | None = None
    if isinstance(ms_level_raw_ml, int):
        ms_level_val_raw = ms_level_raw_ml
    else:
        # mzXML uses 'msLevel' (camelCase)
        ms_level_raw_xml = spectrum.get("msLevel")
        if isinstance(ms_level_raw_xml, int):
            ms_level_val_raw = ms_level_raw_xml
    ms_level = _decode_ms_level(ms_level_val_raw)

    # Polarity
    polarity_str = _extract_polarity_string(spectrum)
    polarity = _decode_polarity(polarity_str)

    # TIC
    tic = _extract_float_or_zero(spectrum, "total ion current")

    # Build structures
    meta = SpectrumMeta(
        source_path=source_path,
        scan_number=scan_number,
        retention_time=retention_time,
        ms_level=ms_level,
        polarity=polarity,
        total_ion_current=tic,
    )

    data = _make_spectrum_data(mz_values, intensities)
    stats = _compute_spectrum_stats(mz_values, intensities)

    return MSSpectrum(meta=meta, data=data, stats=stats)


def _spectrum_to_ms2spectrum(
    spectrum: SpectrumDictProtocol,
    source_path: str,
) -> MS2Spectrum:
    """Convert pyteomics spectrum dict to MS2Spectrum TypedDict.

    Args:
        spectrum: Spectrum dictionary from pyteomics.
        source_path: Path to source file.

    Returns:
        MS2Spectrum TypedDict.

    Raises:
        MzMLReadError: If precursor info not found.
    """
    precursor = _extract_precursor_info(spectrum)
    if precursor is None:
        raise MzMLReadError(source_path, "No precursor info found for MS2 spectrum")

    # Get base spectrum data
    ms_spectrum = _spectrum_to_msspectrum(spectrum, source_path)

    return MS2Spectrum(
        meta=ms_spectrum["meta"],
        precursor=precursor,
        data=ms_spectrum["data"],
        stats=ms_spectrum["stats"],
    )
