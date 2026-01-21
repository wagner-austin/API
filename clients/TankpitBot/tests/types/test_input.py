"""Tests for KeyInput, MouseInput, and ProbeInput TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.types import (
    KeyInput,
    MouseInput,
    ProbeInput,
    decode_key_input,
    decode_mouse_input,
    decode_probe_input,
    encode_key_input,
    encode_mouse_input,
    encode_probe_input,
)

# =============================================================================
# KeyInput Tests
# =============================================================================


def test_encode_key_input() -> None:
    """Test encoding KeyInput to JSON."""
    inp = KeyInput(key="w")
    result = encode_key_input(inp)
    assert result["key"] == "w"


def test_decode_key_input() -> None:
    """Test decoding KeyInput from JSON."""
    data: JSONObject = {"key": "ArrowUp"}
    result = decode_key_input(data)
    assert result["key"] == "ArrowUp"


# =============================================================================
# MouseInput Tests
# =============================================================================


def test_encode_mouse_input() -> None:
    """Test encoding MouseInput to JSON."""
    inp = MouseInput(x=100, y=200, button="left")
    result = encode_mouse_input(inp)
    assert result["x"] == 100
    assert result["y"] == 200
    assert result["button"] == "left"


def test_decode_mouse_input() -> None:
    """Test decoding MouseInput from JSON."""
    data: JSONObject = {"x": 100, "y": 200, "button": "right"}
    result = decode_mouse_input(data)
    assert result["x"] == 100
    assert result["y"] == 200
    assert result["button"] == "right"


def test_decode_mouse_input_middle_button() -> None:
    """Test decoding MouseInput with middle button."""
    data: JSONObject = {"x": 50, "y": 50, "button": "middle"}
    result = decode_mouse_input(data)
    assert result["button"] == "middle"


def test_decode_mouse_input_invalid_button() -> None:
    """Test decoding MouseInput with invalid button raises error."""
    data: JSONObject = {"x": 100, "y": 200, "button": "invalid"}
    with pytest.raises(JSONTypeError, match="must be 'left', 'right', or 'middle'"):
        decode_mouse_input(data)


# =============================================================================
# ProbeInput Tests
# =============================================================================


def test_encode_probe_input_key() -> None:
    """Test encoding ProbeInput with key input."""
    key_inp = KeyInput(key="space")
    inp = ProbeInput(input_type="key", key_input=key_inp, mouse_input=None)
    result = encode_probe_input(inp)
    assert result["input_type"] == "key"
    key_result = result["key_input"]
    assert type(key_result) is dict
    assert key_result["key"] == "space"
    assert result["mouse_input"] is None


def test_encode_probe_input_mouse() -> None:
    """Test encoding ProbeInput with mouse input."""
    mouse_inp = MouseInput(x=300, y=400, button="left")
    inp = ProbeInput(input_type="mouse", key_input=None, mouse_input=mouse_inp)
    result = encode_probe_input(inp)
    assert result["input_type"] == "mouse"
    assert result["key_input"] is None
    mouse_result = result["mouse_input"]
    assert type(mouse_result) is dict
    assert mouse_result["x"] == 300


def test_decode_probe_input_key() -> None:
    """Test decoding ProbeInput with key input."""
    data: JSONObject = {
        "input_type": "key",
        "key_input": {"key": "w"},
        "mouse_input": None,
    }
    result = decode_probe_input(data)
    assert result["input_type"] == "key"
    key_input = result["key_input"]
    assert type(key_input) is dict
    assert key_input["key"] == "w"
    assert result["mouse_input"] is None


def test_decode_probe_input_mouse() -> None:
    """Test decoding ProbeInput with mouse input."""
    data: JSONObject = {
        "input_type": "mouse",
        "key_input": None,
        "mouse_input": {"x": 100, "y": 200, "button": "left"},
    }
    result = decode_probe_input(data)
    assert result["input_type"] == "mouse"
    assert result["key_input"] is None
    mouse_input = result["mouse_input"]
    assert type(mouse_input) is dict
    assert mouse_input["x"] == 100


def test_decode_probe_input_invalid_type() -> None:
    """Test decoding ProbeInput with invalid input type raises error."""
    data: JSONObject = {
        "input_type": "invalid",
        "key_input": None,
        "mouse_input": None,
    }
    with pytest.raises(JSONTypeError, match="must be 'key' or 'mouse'"):
        decode_probe_input(data)


def test_decode_probe_input_invalid_key_input() -> None:
    """Test decoding ProbeInput with invalid key_input type raises error."""
    data: JSONObject = {
        "input_type": "key",
        "key_input": "not an object",
        "mouse_input": None,
    }
    with pytest.raises(JSONTypeError, match="'key_input' must be an object"):
        decode_probe_input(data)


def test_decode_probe_input_invalid_mouse_input() -> None:
    """Test decoding ProbeInput with invalid mouse_input type raises error."""
    data: JSONObject = {
        "input_type": "mouse",
        "key_input": None,
        "mouse_input": "not an object",
    }
    with pytest.raises(JSONTypeError, match="'mouse_input' must be an object"):
        decode_probe_input(data)
