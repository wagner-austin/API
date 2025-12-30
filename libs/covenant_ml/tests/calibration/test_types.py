"""Tests for calibration type definitions and encode/decode functions.

Tests TypedDict validation, encoding, and decoding for calibrator state.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONValue

from covenant_ml.calibration import (
    decode_calibrator_state,
    encode_calibrator_state,
)
from covenant_ml.calibration.testing import make_isotonic_state, make_platt_state
from covenant_ml.calibration.types import (
    CalibratorConfig,
    IsotonicParams,
    IsotonicState,
    PlattParams,
    PlattState,
    _require_bool,
    _require_dict,
    _require_float,
    _require_list_float,
    _require_str,
)

# =============================================================================
# Encode/Decode Round-Trip Tests
# =============================================================================


def test_encode_decode_isotonic_state() -> None:
    """Isotonic state can be encoded and decoded."""
    original = make_isotonic_state(
        x_thresholds=[0.0, 0.25, 0.5, 0.75, 1.0],
        y_values=[0.1, 0.3, 0.5, 0.7, 0.9],
    )

    encoded = encode_calibrator_state(original)
    decoded = decode_calibrator_state(encoded)

    assert decoded["method"] == "isotonic"
    assert decoded["config"]["method"] == "isotonic"
    assert decoded["config"]["clip_proba"] is True
    assert decoded["params"]["X_thresholds"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert decoded["params"]["y_values"] == [0.1, 0.3, 0.5, 0.7, 0.9]


def test_encode_decode_platt_state() -> None:
    """Platt state can be encoded and decoded."""
    original = make_platt_state(slope=2.5, intercept=-1.0)

    encoded = encode_calibrator_state(original)
    decoded = decode_calibrator_state(encoded)

    assert decoded["method"] == "platt"
    assert decoded["config"]["method"] == "platt"
    assert decoded["params"]["A"] == 2.5
    assert decoded["params"]["B"] == -1.0


def test_encode_decode_isotonic_with_custom_config() -> None:
    """Isotonic state with custom config encodes/decodes correctly."""
    original = make_isotonic_state(
        x_thresholds=[0.0, 1.0],
        y_values=[0.0, 1.0],
        clip_proba=False,
        eps=1e-5,
    )

    encoded = encode_calibrator_state(original)
    decoded = decode_calibrator_state(encoded)

    assert decoded["config"]["clip_proba"] is False
    assert decoded["config"]["eps"] == 1e-5


def test_encode_decode_platt_with_custom_config() -> None:
    """Platt state with custom config encodes/decodes correctly."""
    original = make_platt_state(
        slope=1.0,
        intercept=0.0,
        clip_proba=False,
        eps=1e-8,
    )

    encoded = encode_calibrator_state(original)
    decoded = decode_calibrator_state(encoded)

    assert decoded["config"]["clip_proba"] is False
    assert decoded["config"]["eps"] == 1e-8


# =============================================================================
# Encoding Tests
# =============================================================================


def test_encode_isotonic_state_structure() -> None:
    """Encoded isotonic state has correct structure."""
    state = make_isotonic_state()
    encoded = encode_calibrator_state(state)

    # Verify structure by decoding (round-trip validates structure)
    decoded = decode_calibrator_state(encoded)
    assert decoded["method"] == "isotonic"
    assert decoded["config"]["method"] == "isotonic"
    assert decoded["params"]["X_thresholds"] == [0.0, 0.5, 1.0]


def test_encode_platt_state_structure() -> None:
    """Encoded platt state has correct structure."""
    state = make_platt_state()
    encoded = encode_calibrator_state(state)

    # Verify structure by decoding (round-trip validates structure)
    decoded = decode_calibrator_state(encoded)
    assert decoded["method"] == "platt"
    assert decoded["config"]["method"] == "platt"
    assert decoded["params"]["A"] == 1.0


def test_encode_isotonic_state_roundtrip() -> None:
    """Encoded isotonic state can be decoded back correctly."""
    state = make_isotonic_state()
    encoded = encode_calibrator_state(state)

    # Round-trip through encode/decode
    decoded = decode_calibrator_state(encoded)
    assert decoded["method"] == "isotonic"


def test_encode_platt_state_roundtrip() -> None:
    """Encoded platt state can be decoded back correctly."""
    state = make_platt_state()
    encoded = encode_calibrator_state(state)

    # Round-trip through encode/decode
    decoded = decode_calibrator_state(encoded)
    assert decoded["method"] == "platt"


# =============================================================================
# Decoding Validation Tests
# =============================================================================


def test_decode_missing_method_raises() -> None:
    """Decode raises on missing method field."""
    data: dict[str, JSONValue] = {
        "config": {"method": "isotonic", "clip_proba": True, "eps": 1e-10},
        "params": {"X_thresholds": [0.0, 1.0], "y_values": [0.0, 1.0]},
    }

    with pytest.raises(ValueError, match="Field 'method' must be str"):
        decode_calibrator_state(data)


def test_decode_missing_config_raises() -> None:
    """Decode raises on missing config field."""
    data: dict[str, JSONValue] = {
        "method": "isotonic",
        "params": {"X_thresholds": [0.0, 1.0], "y_values": [0.0, 1.0]},
    }

    with pytest.raises(ValueError, match="Field 'config' must be dict"):
        decode_calibrator_state(data)


def test_decode_missing_params_raises() -> None:
    """Decode raises on missing params field."""
    data: dict[str, JSONValue] = {
        "method": "isotonic",
        "config": {"method": "isotonic", "clip_proba": True, "eps": 1e-10},
    }

    with pytest.raises(ValueError, match="Field 'params' must be dict"):
        decode_calibrator_state(data)


def test_decode_invalid_method_raises() -> None:
    """Decode raises on unknown method."""
    data: dict[str, JSONValue] = {
        "method": "unknown",
        "config": {"method": "isotonic", "clip_proba": True, "eps": 1e-10},
        "params": {"X_thresholds": [0.0, 1.0], "y_values": [0.0, 1.0]},
    }

    with pytest.raises(ValueError, match="Unknown calibration method"):
        decode_calibrator_state(data)


def test_decode_invalid_config_method_raises() -> None:
    """Decode raises on invalid config method."""
    data: dict[str, JSONValue] = {
        "method": "isotonic",
        "config": {"method": "invalid", "clip_proba": True, "eps": 1e-10},
        "params": {"X_thresholds": [0.0, 1.0], "y_values": [0.0, 1.0]},
    }

    with pytest.raises(ValueError, match=r"Invalid config\.method"):
        decode_calibrator_state(data)


def test_decode_isotonic_missing_x_thresholds_raises() -> None:
    """Decode raises on missing X_thresholds for isotonic."""
    data: dict[str, JSONValue] = {
        "method": "isotonic",
        "config": {"method": "isotonic", "clip_proba": True, "eps": 1e-10},
        "params": {"y_values": [0.0, 1.0]},
    }

    with pytest.raises(ValueError, match=r"params\.X_thresholds"):
        decode_calibrator_state(data)


def test_decode_isotonic_missing_y_values_raises() -> None:
    """Decode raises on missing y_values for isotonic."""
    data: dict[str, JSONValue] = {
        "method": "isotonic",
        "config": {"method": "isotonic", "clip_proba": True, "eps": 1e-10},
        "params": {"X_thresholds": [0.0, 1.0]},
    }

    with pytest.raises(ValueError, match=r"params\.y_values"):
        decode_calibrator_state(data)


def test_decode_platt_missing_a_raises() -> None:
    """Decode raises on missing A for platt."""
    data: dict[str, JSONValue] = {
        "method": "platt",
        "config": {"method": "platt", "clip_proba": True, "eps": 1e-10},
        "params": {"B": 0.0},
    }

    with pytest.raises(ValueError, match=r"params\.A"):
        decode_calibrator_state(data)


def test_decode_platt_missing_b_raises() -> None:
    """Decode raises on missing B for platt."""
    data: dict[str, JSONValue] = {
        "method": "platt",
        "config": {"method": "platt", "clip_proba": True, "eps": 1e-10},
        "params": {"A": 1.0},
    }

    with pytest.raises(ValueError, match=r"params\.B"):
        decode_calibrator_state(data)


# =============================================================================
# Require Helper Tests
# =============================================================================


def test_require_str_valid() -> None:
    """_require_str returns string for valid input."""
    result = _require_str("hello", "field")
    assert result == "hello"


def test_require_str_invalid() -> None:
    """_require_str raises for non-string input."""
    with pytest.raises(ValueError, match="must be str"):
        _require_str(123, "field")


def test_require_str_none() -> None:
    """_require_str raises for None input."""
    with pytest.raises(ValueError, match="must be str"):
        _require_str(None, "field")


def test_require_float_valid_float() -> None:
    """_require_float returns float for float input."""
    result = _require_float(1.5, "field")
    assert result == 1.5


def test_require_float_valid_int() -> None:
    """_require_float returns float for int input."""
    result = _require_float(2, "field")
    assert result == 2.0


def test_require_float_invalid() -> None:
    """_require_float raises for non-number input."""
    with pytest.raises(ValueError, match="must be number"):
        _require_float("1.5", "field")


def test_require_bool_true() -> None:
    """_require_bool returns True for True input."""
    result = _require_bool(True, "field")
    assert result is True


def test_require_bool_false() -> None:
    """_require_bool returns False for False input."""
    result = _require_bool(False, "field")
    assert result is False


def test_require_bool_invalid() -> None:
    """_require_bool raises for non-bool input."""
    with pytest.raises(ValueError, match="must be bool"):
        _require_bool(1, "field")


def test_require_list_float_valid() -> None:
    """_require_list_float returns list of floats."""
    result = _require_list_float([1, 2.5, 3], "field")
    assert result == [1.0, 2.5, 3.0]


def test_require_list_float_empty() -> None:
    """_require_list_float returns empty list."""
    result = _require_list_float([], "field")
    assert result == []


def test_require_list_float_invalid_not_list() -> None:
    """_require_list_float raises for non-list input."""
    with pytest.raises(ValueError, match="must be list"):
        _require_list_float("not a list", "field")


def test_require_list_float_invalid_item() -> None:
    """_require_list_float raises for non-number item."""
    with pytest.raises(ValueError, match="must be number"):
        _require_list_float([1.0, "two", 3.0], "field")


def test_require_dict_valid() -> None:
    """_require_dict returns dict for dict input."""
    result = _require_dict({"key": "value"}, "field")
    assert result == {"key": "value"}


def test_require_dict_invalid() -> None:
    """_require_dict raises for non-dict input."""
    with pytest.raises(ValueError, match="must be dict"):
        _require_dict("not a dict", "field")


# =============================================================================
# TypedDict Structure Tests
# =============================================================================


def test_calibrator_config_isotonic() -> None:
    """CalibratorConfig works for isotonic method."""
    config: CalibratorConfig = {
        "method": "isotonic",
        "clip_proba": True,
        "eps": 1e-10,
    }
    assert config["method"] == "isotonic"


def test_calibrator_config_platt() -> None:
    """CalibratorConfig works for platt method."""
    config: CalibratorConfig = {
        "method": "platt",
        "clip_proba": False,
        "eps": 1e-8,
    }
    assert config["method"] == "platt"


def test_isotonic_params() -> None:
    """IsotonicParams has required fields."""
    params: IsotonicParams = {
        "X_thresholds": [0.0, 0.5, 1.0],
        "y_values": [0.1, 0.5, 0.9],
    }
    assert len(params["X_thresholds"]) == 3
    assert len(params["y_values"]) == 3


def test_platt_params() -> None:
    """PlattParams has required fields."""
    params: PlattParams = {
        "A": 1.5,
        "B": -0.5,
    }
    assert params["A"] == 1.5
    assert params["B"] == -0.5


def test_isotonic_state() -> None:
    """IsotonicState has all required fields."""
    state: IsotonicState = {
        "method": "isotonic",
        "config": {
            "method": "isotonic",
            "clip_proba": True,
            "eps": 1e-10,
        },
        "params": {
            "X_thresholds": [0.0, 1.0],
            "y_values": [0.0, 1.0],
        },
    }
    assert state["method"] == "isotonic"


def test_platt_state() -> None:
    """PlattState has all required fields."""
    state: PlattState = {
        "method": "platt",
        "config": {
            "method": "platt",
            "clip_proba": True,
            "eps": 1e-10,
        },
        "params": {
            "A": 1.0,
            "B": 0.0,
        },
    }
    assert state["method"] == "platt"
