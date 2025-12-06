"""Decoder functions for MGF peak list data via pyteomics.

Converts pyteomics MGF spectrum dictionaries to typed structures.
"""

from __future__ import annotations

from instrument_io._exceptions import DecodingError
from instrument_io.types.common import MSLevel, Polarity
from instrument_io.types.spectrum import (
    PrecursorInfo,
    SpectrumData,
    SpectrumMeta,
    SpectrumStats,
)


def _decode_mgf_title(
    params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None],
) -> str:
    """Extract title from MGF params.

    Args:
        params: MGF spectrum params dict.

    Returns:
        Spectrum title string.
    """
    title = params.get("title")
    if title is None:
        return ""
    if isinstance(title, str):
        return title
    return str(title)


def _decode_pepmass(
    pepmass: str | float | int | list[int] | tuple[float | None, ...] | None,
) -> tuple[float, float | None]:
    """Decode pepmass value to (mz, intensity) tuple.

    Args:
        pepmass: Raw pepmass value from params. pyteomics returns tuples
                 like (mz,) or (mz, intensity) where intensity may be None.

    Returns:
        Tuple of (precursor_mz, precursor_intensity).

    Raises:
        DecodingError: If pepmass is invalid.
    """
    if pepmass is None:
        raise DecodingError("precursor", "Missing pepmass in MGF spectrum")

    if isinstance(pepmass, tuple):
        if len(pepmass) == 0:
            raise DecodingError("precursor", "Empty pepmass tuple")
        first_element = pepmass[0]
        if first_element is None:
            raise DecodingError("precursor", "Missing m/z in pepmass tuple")
        mz = float(first_element)
        # pyteomics may return (mz, None) when intensity is absent
        intensity: float | None = None
        if len(pepmass) > 1:
            second_element = pepmass[1]
            if second_element is not None:
                intensity = float(second_element)
        return mz, intensity

    if isinstance(pepmass, (int, float)):
        return float(pepmass), None

    raise DecodingError("precursor", f"Invalid pepmass type: {type(pepmass).__name__}")


def _decode_charge_value(
    charge_raw: str | float | int | list[int] | tuple[float | None, ...] | None,
) -> int | None:
    """Decode charge value from various formats.

    Args:
        charge_raw: Raw charge value from params.

    Returns:
        Charge as int or None if not present/parseable.
    """
    if charge_raw is None:
        return None

    if isinstance(charge_raw, int):
        return charge_raw

    if isinstance(charge_raw, list) and len(charge_raw) > 0:
        return charge_raw[0]

    if isinstance(charge_raw, str):
        charge_str = charge_raw.replace("+", "").replace("-", "")
        if charge_str.isdigit():
            charge = int(charge_str)
            return -charge if "-" in charge_raw else charge

    return None


def _decode_mgf_precursor(
    params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None],
) -> PrecursorInfo:
    """Extract precursor information from MGF params.

    Args:
        params: MGF spectrum params dict containing pepmass, charge.

    Returns:
        PrecursorInfo TypedDict.

    Raises:
        DecodingError: If pepmass is missing or invalid.
    """
    precursor_mz, precursor_intensity = _decode_pepmass(params.get("pepmass"))
    charge = _decode_charge_value(params.get("charge"))

    return PrecursorInfo(
        mz=precursor_mz,
        charge=charge,
        intensity=precursor_intensity,
        isolation_window=None,
    )


def _decode_mgf_polarity(
    params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None],
) -> Polarity:
    """Extract polarity from MGF params based on charge sign.

    Args:
        params: MGF spectrum params dict.

    Returns:
        Polarity literal based on decoded charge.
    """
    # Reuse _decode_charge_value to get the numeric charge
    charge = _decode_charge_value(params.get("charge"))
    if charge is None:
        return "unknown"
    if charge > 0:
        return "positive"
    if charge < 0:
        return "negative"
    return "unknown"


def _decode_mgf_scan_number(
    params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None],
    index: int,
) -> int:
    """Extract scan number from MGF params or use index.

    Args:
        params: MGF spectrum params dict.
        index: 0-based index of spectrum in file.

    Returns:
        Scan number (1-based).
    """
    # Try to extract from title (e.g., "Scan 1234" or "scan=1234")
    title = params.get("title")
    if title is not None and isinstance(title, str):
        title_lower = title.lower()
        if "scan=" in title_lower:
            parts = title_lower.split("scan=")
            num_part = parts[1].split()[0] if len(parts) > 1 else ""
            if num_part.isdigit():
                return int(num_part)
        if "scan " in title_lower:
            parts = title_lower.split("scan ")
            num_part = parts[1].split()[0] if len(parts) > 1 else ""
            if num_part.isdigit():
                return int(num_part)

    # Try scans parameter
    scans = params.get("scans")
    if scans is not None:
        if isinstance(scans, int):
            return scans
        if isinstance(scans, str) and scans.isdigit():
            return int(scans)

    # Fallback to 1-based index
    return index + 1


def _is_numeric_string(value: str) -> bool:
    """Check if string represents a valid float.

    Args:
        value: String to check.

    Returns:
        True if string can be converted to float.
    """
    # Handle negative numbers and decimals
    stripped = value.strip()
    if not stripped:
        return False
    # Remove leading minus sign for checking
    check_str = stripped[1:] if stripped.startswith("-") else stripped
    # Check for valid float pattern: digits with optional single decimal point
    parts = check_str.split(".")
    if len(parts) > 2:
        return False
    return all(part.isdigit() for part in parts if part)


def _decode_mgf_retention_time(
    params: dict[str, str | float | int | list[int] | tuple[float | None, ...] | None],
) -> float:
    """Extract retention time from MGF params if present.

    Args:
        params: MGF spectrum params dict.

    Returns:
        Retention time in minutes (0.0 if not present).
    """
    rtinseconds = params.get("rtinseconds")
    if rtinseconds is not None:
        if isinstance(rtinseconds, (int, float)):
            return float(rtinseconds) / 60.0
        if isinstance(rtinseconds, str) and _is_numeric_string(rtinseconds):
            return float(rtinseconds) / 60.0

    rt = params.get("retentiontime")
    if rt is not None:
        if isinstance(rt, (int, float)):
            return float(rt)
        if isinstance(rt, str) and _is_numeric_string(rt):
            return float(rt)

    return 0.0


def _compute_mgf_spectrum_stats(
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


def _make_mgf_spectrum_meta(
    source_path: str,
    scan_number: int,
    retention_time: float,
    polarity: Polarity,
    total_ion_current: float,
) -> SpectrumMeta:
    """Create SpectrumMeta for MGF spectrum.

    Args:
        source_path: Path to source file.
        scan_number: Scan number (1-based).
        retention_time: Retention time in minutes.
        polarity: Ion polarity.
        total_ion_current: TIC value.

    Returns:
        SpectrumMeta TypedDict.
    """
    ms_level: MSLevel = 2  # MGF is always MS/MS data
    return SpectrumMeta(
        source_path=source_path,
        scan_number=scan_number,
        retention_time=retention_time,
        ms_level=ms_level,
        polarity=polarity,
        total_ion_current=total_ion_current,
    )


def _make_mgf_spectrum_data(
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


__all__ = [
    "_compute_mgf_spectrum_stats",
    "_decode_mgf_polarity",
    "_decode_mgf_precursor",
    "_decode_mgf_retention_time",
    "_decode_mgf_scan_number",
    "_decode_mgf_title",
    "_make_mgf_spectrum_data",
    "_make_mgf_spectrum_meta",
]
