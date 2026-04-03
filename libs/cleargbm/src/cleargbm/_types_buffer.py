"""Buffer serialization type definitions for ClearGBM.

Provides FloatBufferData, IntBufferData, and HistogramBufferData TypedDicts
with their encode/decode functions for JSON persistence of buffer contents.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

from typing import TypedDict

from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    _require_int,
    require_positive_int,
)

# =============================================================================
# Buffer Serialization Types
# =============================================================================


class FloatBufferData(TypedDict):
    """Serialized FloatBuffer for JSON persistence.

    Args:
        values: Tuple of float values.
        size: Number of elements.
    """

    values: tuple[float, ...]
    size: int


def encode_float_buffer_data(
    values: tuple[float, ...],
    size: int,
) -> JSONDict:
    """Encode FloatBuffer data to JSON-serializable dict.

    Args:
        values: Buffer values as tuple.
        size: Buffer size.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "values": list(values),
        "size": size,
    }


def decode_float_buffer_data(raw: JSONDict) -> FloatBufferData:
    """Decode raw dict to FloatBufferData.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated FloatBufferData.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If validation fails.
    """
    size = require_positive_int(_require_int(raw, "size"), "size")

    values_raw = raw["values"]
    if not isinstance(values_raw, list):
        raise JSONTypeError(f"values must be list, got {type(values_raw).__name__}")

    values: list[float] = []
    for i, val in enumerate(values_raw):
        if isinstance(val, bool):
            raise JSONTypeError(f"values[{i}] must be float, got bool")
        if isinstance(val, int):
            values.append(float(val))
        elif isinstance(val, float):
            values.append(val)
        else:
            raise JSONTypeError(f"values[{i}] must be float, got {type(val).__name__}")

    if len(values) != size:
        raise ValueError(f"values length {len(values)} != size {size}")

    return FloatBufferData(values=tuple(values), size=size)


class IntBufferData(TypedDict):
    """Serialized IntBuffer for JSON persistence.

    Args:
        values: Tuple of int values.
        size: Number of elements.
    """

    values: tuple[int, ...]
    size: int


def encode_int_buffer_data(values: tuple[int, ...], size: int) -> JSONDict:
    """Encode IntBuffer data to JSON-serializable dict.

    Args:
        values: Buffer values as tuple.
        size: Buffer size.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "values": list(values),
        "size": size,
    }


def decode_int_buffer_data(raw: JSONDict) -> IntBufferData:
    """Decode raw dict to IntBufferData.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated IntBufferData.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If validation fails.
    """
    size = require_positive_int(_require_int(raw, "size"), "size")

    values_raw = raw["values"]
    if not isinstance(values_raw, list):
        raise JSONTypeError(f"values must be list, got {type(values_raw).__name__}")

    values: list[int] = []
    for i, val in enumerate(values_raw):
        if not isinstance(val, int) or isinstance(val, bool):
            raise JSONTypeError(f"values[{i}] must be int, got {type(val).__name__}")
        values.append(val)

    if len(values) != size:
        raise ValueError(f"values length {len(values)} != size {size}")

    return IntBufferData(values=tuple(values), size=size)


class HistogramBufferData(TypedDict):
    """Serialized HistogramBuffer for JSON persistence.

    Args:
        gradient_sums: Gradient sum per bin.
        hessian_sums: Hessian sum per bin.
        counts: Sample count per bin.
        n_bins: Number of bins.
    """

    gradient_sums: tuple[float, ...]
    hessian_sums: tuple[float, ...]
    counts: tuple[int, ...]
    n_bins: int


def encode_histogram_buffer_data(
    gradient_sums: tuple[float, ...],
    hessian_sums: tuple[float, ...],
    counts: tuple[int, ...],
    n_bins: int,
) -> JSONDict:
    """Encode HistogramBuffer data to JSON-serializable dict.

    Args:
        gradient_sums: Gradient sums per bin.
        hessian_sums: Hessian sums per bin.
        counts: Sample counts per bin.
        n_bins: Number of bins.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "gradient_sums": list(gradient_sums),
        "hessian_sums": list(hessian_sums),
        "counts": list(counts),
        "n_bins": n_bins,
    }


def _require_float_list(raw: JSONDict, key: str) -> list[float]:
    """Extract and validate a list of floats from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        List of float values.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value has wrong type.
    """
    values_raw = raw[key]
    if not isinstance(values_raw, list):
        raise JSONTypeError(f"{key} must be list, got {type(values_raw).__name__}")

    result: list[float] = []
    for i, val in enumerate(values_raw):
        if isinstance(val, bool):
            raise JSONTypeError(f"{key}[{i}] must be float, got bool")
        if isinstance(val, int):
            result.append(float(val))
        elif isinstance(val, float):
            result.append(val)
        else:
            raise JSONTypeError(f"{key}[{i}] must be float, got {type(val).__name__}")
    return result


def _require_int_list(raw: JSONDict, key: str) -> list[int]:
    """Extract and validate a list of ints from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        List of int values.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value has wrong type.
    """
    values_raw = raw[key]
    if not isinstance(values_raw, list):
        raise JSONTypeError(f"{key} must be list, got {type(values_raw).__name__}")

    result: list[int] = []
    for i, val in enumerate(values_raw):
        if not isinstance(val, int) or isinstance(val, bool):
            raise JSONTypeError(f"{key}[{i}] must be int, got {type(val).__name__}")
        result.append(val)
    return result


def decode_histogram_buffer_data(raw: JSONDict) -> HistogramBufferData:
    """Decode raw dict to HistogramBufferData.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated HistogramBufferData.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If validation fails.
    """
    n_bins = require_positive_int(_require_int(raw, "n_bins"), "n_bins")
    gradient_sums = _require_float_list(raw, "gradient_sums")
    hessian_sums = _require_float_list(raw, "hessian_sums")
    counts = _require_int_list(raw, "counts")

    # Validate lengths
    if len(gradient_sums) != n_bins:
        raise ValueError(f"gradient_sums length {len(gradient_sums)} != n_bins {n_bins}")
    if len(hessian_sums) != n_bins:
        raise ValueError(f"hessian_sums length {len(hessian_sums)} != n_bins {n_bins}")
    if len(counts) != n_bins:
        raise ValueError(f"counts length {len(counts)} != n_bins {n_bins}")

    return HistogramBufferData(
        gradient_sums=tuple(gradient_sums),
        hessian_sums=tuple(hessian_sums),
        counts=tuple(counts),
        n_bins=n_bins,
    )


__all__ = [
    "FloatBufferData",
    "HistogramBufferData",
    "IntBufferData",
    "decode_float_buffer_data",
    "decode_histogram_buffer_data",
    "decode_int_buffer_data",
    "encode_float_buffer_data",
    "encode_histogram_buffer_data",
    "encode_int_buffer_data",
]
