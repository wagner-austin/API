"""Translation schemas for grandma-api.

Provides TypedDict schemas with encode/decode/require_* validation for
the /translate endpoint responses.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_str,
)
from typing_extensions import TypedDict


class TranslationResponse(TypedDict):
    """Response from audio translation endpoint.

    Attributes:
        text: The translated English text.
        detected_language: ISO 639-1 code of detected source language.
        source_text: Original transcription in source language.
        confidence: Confidence score for language detection (0.0-1.0).
    """

    text: str
    detected_language: str
    source_text: str
    confidence: float


def encode_translation_response(response: TranslationResponse) -> JSONObject:
    """Encode TranslationResponse to JSON-compatible dict.

    Args:
        response: The response to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "text": response["text"],
        "detected_language": response["detected_language"],
        "source_text": response["source_text"],
        "confidence": response["confidence"],
    }


def decode_translation_response(obj: JSONObject) -> TranslationResponse:
    """Decode JSON object to TranslationResponse with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TranslationResponse.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    text = require_str(obj, "text")
    detected_language = require_str(obj, "detected_language")
    source_text = require_str(obj, "source_text")
    confidence = require_float(obj, "confidence")

    return TranslationResponse(
        text=text,
        detected_language=detected_language,
        source_text=source_text,
        confidence=confidence,
    )


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
