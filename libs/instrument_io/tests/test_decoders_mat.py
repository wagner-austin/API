"""Unit tests for MATLAB decoder helpers.

Exercises decoding using the real MAT fixture (no mocks/fakes),
and validates scalar-path handling for the pure function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypeGuard

from instrument_io._decoders.mat import _decode_mat_array, _decode_mat_dict, _is_metadata_key
from instrument_io._protocols.scipy_io import MATDict, NdarrayProtocol, _load_mat


def _is_ndarray_protocol(
    value: NdarrayProtocol | str | int | float,
) -> TypeGuard[NdarrayProtocol]:
    """Type guard to narrow values that conform to NdarrayProtocol."""
    return hasattr(value, "tolist") and hasattr(value, "shape") and hasattr(value, "dtype")


def test_is_metadata_key_true_and_false() -> None:
    """_is_metadata_key returns True only for __name__ style keys."""
    assert _is_metadata_key("__header__") is True
    assert _is_metadata_key("__version__") is True
    assert _is_metadata_key("matrix") is False
    assert _is_metadata_key("header__") is False
    assert _is_metadata_key("__header") is False


def test_decode_mat_array_uses_tolist() -> None:
    """_decode_mat_array works with a real ndarray from the MAT fixture."""
    fixtures = Path(__file__).parent / "fixtures"
    mat_path = fixtures / "sample.mat"
    mat_dict = _load_mat(mat_path)
    # Pick the first non-metadata entry
    key = next(k for k in mat_dict if not _is_metadata_key(k))
    value = mat_dict[key]
    assert _is_ndarray_protocol(value)
    decoded = _decode_mat_array(value)
    assert decoded == value.tolist()


def test_decode_mat_dict_converts_arrays_and_skips_metadata_from_fixture() -> None:
    """_decode_mat_dict converts ndarrays and drops metadata keys using real MAT file."""
    fixtures = Path(__file__).parent / "fixtures"
    mat_path = fixtures / "sample.mat"
    mat_dict = _load_mat(mat_path)

    result = _decode_mat_dict(mat_dict)

    # Ensure metadata keys were removed
    assert all(not _is_metadata_key(k) for k in result)

    # Ensure at least one array variable is present and converted to list
    assert any(isinstance(v, list) for v in result.values())


def test_decode_mat_dict_scalar_path() -> None:
    """_decode_mat_dict preserves plain scalars as-is for allowed types."""
    mat_dict: MATDict = {
        "scalar_int": 7,
        "scalar_float": 2.5,
        "scalar_text": "ok",
        "__header__": "skip-me",
    }
    result = _decode_mat_dict(mat_dict)
    expected_keys: Final[set[str]] = {"scalar_int", "scalar_float", "scalar_text"}
    assert set(result.keys()) == expected_keys
    assert result["scalar_int"] == 7
    assert result["scalar_float"] == 2.5
    assert result["scalar_text"] == "ok"
