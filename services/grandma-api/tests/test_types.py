"""Tests for grandma_api.types module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from grandma_api.types import (
    TranslationResponse,
    decode_translation_response,
    encode_translation_response,
    require_translation_response,
)


def test_encode_translation_response() -> None:
    """Test encoding TranslationResponse to JSON."""
    result = TranslationResponse(text="Hello from grandmother")
    encoded = encode_translation_response(result)
    assert encoded == {"text": "Hello from grandmother"}


def test_decode_translation_response() -> None:
    """Test decoding JSON to TranslationResponse."""
    obj: JSONObject = {"text": "Hello from grandmother"}
    decoded = decode_translation_response(obj)
    assert decoded["text"] == "Hello from grandmother"


def test_decode_translation_response_missing_text() -> None:
    """Test decode raises when text field is missing."""
    obj: JSONObject = {"other": "value"}
    with pytest.raises(JSONTypeError, match="Missing required field 'text'"):
        decode_translation_response(obj)


def test_decode_translation_response_wrong_type() -> None:
    """Test decode raises when text field has wrong type."""
    obj: JSONObject = {"text": 123}
    with pytest.raises(JSONTypeError, match="Field 'text' must be a string"):
        decode_translation_response(obj)


def test_require_translation_response() -> None:
    """Test require_translation_response with valid dict."""
    obj: JSONValue = {"text": "Hello"}
    result = require_translation_response(obj)
    assert result["text"] == "Hello"


def test_require_translation_response_not_dict() -> None:
    """Test require_translation_response raises when not a dict."""
    value: JSONValue = "not a dict"
    with pytest.raises(JSONTypeError, match="Expected object"):
        require_translation_response(value)


def test_require_translation_response_list() -> None:
    """Test require_translation_response raises when given a list."""
    value: JSONValue = []
    with pytest.raises(JSONTypeError, match="Expected object"):
        require_translation_response(value)
