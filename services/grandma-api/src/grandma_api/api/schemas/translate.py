"""Translation schemas for grandma-api.

Provides TypedDict schemas with encode/decode/require_* validation for
the /translate endpoint responses.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_str,
)
from typing_extensions import TypedDict


class TranslationResponse(TypedDict):
    """Response from audio translation endpoint.

    Attributes:
        text: The translated English text from Vietnamese audio.
    """

    text: str


def encode_translation_response(response: TranslationResponse) -> JSONObject:
    """Encode TranslationResponse to JSON-compatible dict.

    Args:
        response: The response to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {"text": response["text"]}


def decode_translation_response(obj: JSONObject) -> TranslationResponse:
    """Decode JSON object to TranslationResponse with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TranslationResponse.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    return TranslationResponse(text=require_str(obj, "text"))


def require_translation_response(obj: JSONValue) -> TranslationResponse:
    """Validate and convert JSONValue to TranslationResponse.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated TranslationResponse.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_translation_response(obj)


__all__ = [
    "TranslationResponse",
    "decode_translation_response",
    "encode_translation_response",
    "require_translation_response",
]
