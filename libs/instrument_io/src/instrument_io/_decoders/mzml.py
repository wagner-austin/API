"""Decoder functions for mzML/mzXML data via pyteomics.

Converts pyteomics spectrum dictionaries to typed structures.
"""

from __future__ import annotations

from instrument_io._exceptions import DecodingError
from instrument_io.types.common import MSLevel, Polarity
from instrument_io.types.spectrum import SpectrumData, SpectrumStats


def _decode_mz_array(arr_list: list[float]) -> list[float]:
    """Decode m/z array from ndarray.tolist() result.

    Args:
        arr_list: Result of mz_array.tolist().

    Returns:
        List of m/z values.
    """
    if not arr_list:
        return []
    return arr_list


def _decode_intensity_array(arr_list: list[float]) -> list[float]:
    """Decode intensity array from ndarray.tolist() result.

    Args:
        arr_list: Result of intensity_array.tolist().

    Returns:
        List of intensity values.
    """
    if not arr_list:
        return []
    return arr_list


def _decode_polarity(scan_polarity: str | None) -> Polarity:
    """Decode polarity from mzML scan attribute.

    Args:
        scan_polarity: Polarity string from mzML.

    Returns:
        Polarity literal ("positive", "negative", or "unknown").
    """
    if scan_polarity is None:
        return "unknown"

    pol_lower = scan_polarity.lower()
    if "positive" in pol_lower or "+" in pol_lower:
        return "positive"
    if "negative" in pol_lower or "-" in pol_lower:
        return "negative"

    return "unknown"


def _decode_ms_level(level_raw: int | float | str | None) -> MSLevel:
    """Decode MS level from mzML attribute.

    Args:
        level_raw: MS level value from mzML.

    Returns:
        MSLevel literal (1, 2, or 3).

    Raises:
        DecodingError: If level is not 1, 2, or 3.
    """
    if level_raw is None:
        return 1

    level_int = int(level_raw) if isinstance(level_raw, (str, float)) else level_raw

    if level_int == 1:
        return 1
    if level_int == 2:
        return 2
    if level_int == 3:
        return 3

    raise DecodingError("ms_level", f"Invalid MS level: {level_int}")


def _decode_retention_time(rt_value: float | int | str | None) -> float:
    """Decode retention time from mzML attribute.

    Args:
        rt_value: Retention time value (may be in seconds).

    Returns:
        Retention time in minutes.
    """
    if rt_value is None:
        return 0.0

    rt_float = float(rt_value)

    # If value seems to be in seconds (> 100), convert to minutes
    if rt_float > 100:
        return rt_float / 60.0

    return rt_float


def _decode_scan_number(scan_str: str | int | None) -> int:
    """Decode scan number from mzML spectrum ID.

    Handles multiple formats:
    - "scan=1234" (ProteoWizard style)
    - "sample=1 period=1 cycle=22 experiment=1" (Waters style, uses cycle)
    - "controllerType=0 controllerNumber=1 scan=1234" (Thermo style)
    - "1234" or "S1234" (simple numeric)

    Args:
        scan_str: Scan identifier from mzML.

    Returns:
        Integer scan number.

    Raises:
        DecodingError: If scan_str cannot be parsed as an integer.
    """
    if scan_str is None:
        return 0

    if isinstance(scan_str, int):
        return scan_str

    # Try to extract "scan=" value first
    if "scan=" in scan_str:
        parts = scan_str.split("scan=")
        num_part = parts[1].split()[0]
        return int(num_part)

    # Try to extract "cycle=" value (Waters format)
    if "cycle=" in scan_str:
        parts = scan_str.split("cycle=")
        num_part = parts[1].split()[0]
        return int(num_part)

    # Try direct numeric conversion
    if scan_str.isdigit():
        return int(scan_str)

    # Handle "S1234" format
    if scan_str.startswith("S") and scan_str[1:].isdigit():
        return int(scan_str[1:])

    # Handle negative numbers
    if scan_str.startswith("-") and scan_str[1:].isdigit():
        return int(scan_str)

    # If nothing works, try to extract first number from string
    import re

    match = re.search(r"\d+", scan_str)
    if match is not None:
        return int(match.group())

    raise DecodingError("scan_number", f"Cannot parse '{scan_str}' as scan number")


def _compute_spectrum_stats(
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


def _make_spectrum_data(
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
    "_compute_spectrum_stats",
    "_decode_intensity_array",
    "_decode_ms_level",
    "_decode_mz_array",
    "_decode_polarity",
    "_decode_retention_time",
    "_decode_scan_number",
    "_make_spectrum_data",
]
