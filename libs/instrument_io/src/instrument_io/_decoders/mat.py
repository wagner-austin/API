"""Decoder functions for MATLAB .mat files.

Converts scipy.io.loadmat results to typed Python structures.
"""

from __future__ import annotations

from instrument_io._protocols.scipy_io import MATDict, NdarrayProtocol, PythonValue


def _is_metadata_key(key: str) -> bool:
    """Check if dictionary key is MATLAB metadata (starts/ends with __).

    Args:
        key: Dictionary key from loadmat result.

    Returns:
        True if key is metadata (e.g., __header__, __version__, __globals__).
    """
    return key.startswith("__") and key.endswith("__")


def _decode_mat_array(arr: NdarrayProtocol) -> PythonValue:
    """Convert numpy array to Python list.

    Args:
        arr: Numpy array from scipy.io.loadmat.

    Returns:
        Python list representation of array.
    """
    return arr.tolist()


def _decode_mat_dict(mat_dict: MATDict) -> dict[str, PythonValue | str | int | float]:
    """Decode MATLAB dictionary, converting arrays to lists.

    Args:
        mat_dict: Dictionary from scipy.io.loadmat.

    Returns:
        Dictionary with numpy arrays converted to Python lists.
        Excludes MATLAB metadata keys (starting/ending with __).
    """
    result: dict[str, PythonValue | str | int | float] = {}

    for key, value in mat_dict.items():
        # Skip metadata keys
        if _is_metadata_key(key):
            continue

        # Convert ndarray to list
        if isinstance(value, (str, int, float)):
            result[key] = value
        else:
            # value is NdarrayProtocol by type narrowing
            result[key] = _decode_mat_array(value)

    return result


__all__ = [
    "_decode_mat_array",
    "_decode_mat_dict",
    "_is_metadata_key",
]
