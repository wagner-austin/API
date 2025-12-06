"""Decoder functions for Agilent data via rainbow-api.

Converts rainbow DataFile/DataDirectory data to typed structures.
Pure Python implementations for type safety - no numpy operations.
"""

from __future__ import annotations

from typing import TypeGuard

from instrument_io._exceptions import DecodingError
from instrument_io.types.chromatogram import ChromatogramData, ChromatogramStats
from instrument_io.types.common import SignalType

# Type alias for the union returned by NdArrayProtocol.tolist()
_ToListResult = list[float] | list[list[float]] | list[int] | list[list[int]]

# Narrowed types for 1D and 2D variants
_ToListResult1D = list[float] | list[int]
_ToListResult2D = list[list[float]] | list[list[int]]


def _is_1d_list(raw: _ToListResult) -> TypeGuard[_ToListResult1D]:
    """TypeGuard: check if tolist result is 1D (list of scalars)."""
    if not raw:
        return False
    return not isinstance(raw[0], list)


def _is_2d_list(raw: _ToListResult) -> TypeGuard[_ToListResult2D]:
    """TypeGuard: check if tolist result is 2D (list of lists)."""
    if not raw:
        return False
    return isinstance(raw[0], list)


def _narrow_tolist_1d(raw: _ToListResult) -> list[float]:
    """Narrow tolist() result to 1D float list.

    Args:
        raw: Result of ndarray.tolist() known to be 1D.

    Returns:
        list[float] after validation.

    Raises:
        DecodingError: If data is not 1D or contains nested lists.
    """
    if not raw:
        raise DecodingError("narrow_1d", "Empty array")

    if not _is_1d_list(raw):
        raise DecodingError("narrow_1d", "Expected 1D array but got nested list")

    # TypeGuard narrowed raw to _ToListResult1D (list[float] | list[int])
    return [float(v) for v in raw]


def _narrow_tolist_2d(raw: _ToListResult) -> list[list[float]]:
    """Narrow tolist() result to 2D float list.

    Args:
        raw: Result of ndarray.tolist() known to be 2D.

    Returns:
        list[list[float]] after validation.

    Raises:
        DecodingError: If data is not 2D.
    """
    if not raw:
        raise DecodingError("narrow_2d", "Empty array")

    if not _is_2d_list(raw):
        raise DecodingError("narrow_2d", "Expected 2D array but got flat list")

    # TypeGuard narrowed raw to _ToListResult2D (list[list[float]] | list[list[int]])
    return [[float(v) for v in row] for row in raw]


def _decode_retention_times(xlabels_list: list[float]) -> list[float]:
    """Decode retention times from ndarray.tolist() result.

    Args:
        xlabels_list: Result of xlabels.tolist().

    Returns:
        List of retention times in minutes.

    Raises:
        DecodingError: If retention times array is empty.
    """
    if not xlabels_list:
        raise DecodingError("retention_times", "Empty retention times array")
    return xlabels_list


def _decode_intensities_1d(data_list: list[float]) -> list[float]:
    """Decode 1D intensity array.

    Args:
        data_list: 1D list of intensities (int values are converted to float).

    Returns:
        List of intensity values as floats.

    Raises:
        DecodingError: If array is empty.
    """
    if not data_list:
        raise DecodingError("intensities", "Empty intensities array")

    # Type signature guarantees list[float]; int->float conversion is implicit
    return [float(val) for val in data_list]


def _decode_intensities_2d(data_list: list[list[float]], row_index: int = 0) -> list[float]:
    """Decode specific row from 2D intensity array.

    Args:
        data_list: 2D list of intensities [channel][time].
        row_index: Row/channel index to extract.

    Returns:
        List of intensity values for specified row.

    Raises:
        DecodingError: If array is empty or index out of range.
    """
    if not data_list:
        raise DecodingError("intensities", "Empty 2D intensities array")

    if row_index >= len(data_list):
        raise DecodingError(
            "intensities",
            f"Row index {row_index} out of range for array with {len(data_list)} rows",
        )

    row = data_list[row_index]
    return _decode_intensities_1d(row)


def _sum_2d_to_tic(data_list: list[list[float]]) -> list[float]:
    """Sum 2D data across all channels to produce TIC.

    Args:
        data_list: 2D list [channel][time].

    Returns:
        1D list with summed intensities across channels.

    Raises:
        DecodingError: If array is empty or rows have different lengths.
    """
    if not data_list:
        raise DecodingError("sum_2d", "Empty 2D array")

    if not data_list[0]:
        raise DecodingError("sum_2d", "Empty first row")

    num_points = len(data_list[0])
    result: list[float] = [0.0] * num_points

    for row_idx, row in enumerate(data_list):
        if len(row) != num_points:
            raise DecodingError(
                "sum_2d",
                f"Row {row_idx} has {len(row)} points, expected {num_points}",
            )
        for i, val in enumerate(row):
            result[i] += float(val)

    return result


def _decode_signal_type(detector: str) -> SignalType:
    """Decode signal type from detector name string.

    Args:
        detector: Detector name from rainbow DataFile.

    Returns:
        Appropriate SignalType literal.
    """
    detector_lower = detector.lower()

    if "tic" in detector_lower or "total" in detector_lower:
        return "TIC"
    if "eic" in detector_lower or "extracted" in detector_lower:
        return "EIC"
    if "dad" in detector_lower or "diode" in detector_lower:
        return "DAD"
    if "uv" in detector_lower:
        return "UV"
    if "fid" in detector_lower:
        return "FID"
    if "ms" in detector_lower:
        return "MS"

    # Default to MS for unknown detectors in MS data
    return "MS"


def _compute_chromatogram_stats(
    retention_times: list[float],
    intensities: list[float],
) -> ChromatogramStats:
    """Compute statistics from chromatogram data.

    Pure Python implementation for type safety.

    Args:
        retention_times: List of time points.
        intensities: List of intensity values.

    Returns:
        ChromatogramStats TypedDict.

    Raises:
        DecodingError: If data is empty or mismatched lengths.
    """
    if not retention_times or not intensities:
        raise DecodingError("stats", "Empty data for stats computation")

    if len(retention_times) != len(intensities):
        raise DecodingError(
            "stats",
            f"Mismatched lengths: {len(retention_times)} times vs {len(intensities)} intensities",
        )

    num_points = len(retention_times)
    rt_min = min(retention_times)
    rt_max = max(retention_times)
    rt_step_mean = (rt_max - rt_min) / (num_points - 1) if num_points > 1 else 0.0

    intensity_min = min(intensities)
    intensity_max = max(intensities)
    intensity_mean = sum(intensities) / num_points
    intensity_p99 = _percentile(intensities, 0.99)

    return ChromatogramStats(
        num_points=num_points,
        rt_min=rt_min,
        rt_max=rt_max,
        rt_step_mean=rt_step_mean,
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        intensity_mean=intensity_mean,
        intensity_p99=intensity_p99,
    )


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile using linear interpolation.

    Args:
        values: List of numeric values.
        p: Percentile (0.0 to 1.0).

    Returns:
        Interpolated percentile value.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = p * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def _make_chromatogram_data(
    retention_times: list[float],
    intensities: list[float],
) -> ChromatogramData:
    """Create ChromatogramData TypedDict.

    Args:
        retention_times: List of time points.
        intensities: List of intensity values.

    Returns:
        ChromatogramData TypedDict.
    """
    return ChromatogramData(
        retention_times=retention_times,
        intensities=intensities,
    )


__all__ = [
    "_compute_chromatogram_stats",
    "_decode_intensities_1d",
    "_decode_intensities_2d",
    "_decode_retention_times",
    "_decode_signal_type",
    "_is_1d_list",
    "_is_2d_list",
    "_make_chromatogram_data",
    "_narrow_tolist_1d",
    "_narrow_tolist_2d",
    "_percentile",
    "_sum_2d_to_tic",
]
