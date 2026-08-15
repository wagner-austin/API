"""types: VerboseSegment and related definitions."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_str,
    require_float,
    require_list,
    require_str,
)
from typing_extensions import TypedDict


class VerboseSegment(TypedDict):
    """A segment from OpenAI Whisper verbose response."""

    text: str
    start: float
    end: float


class VerboseResponse(TypedDict):
    """OpenAI Whisper verbose_json response format."""

    text: str
    language: str | None
    segments: list[VerboseSegment]


def encode_verbose_segment(segment: VerboseSegment) -> JSONObject:
    """Encode VerboseSegment to JSON-compatible dict.

    Args:
        segment: The segment to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "text": segment["text"],
        "start": segment["start"],
        "end": segment["end"],
    }


def decode_verbose_segment(obj: JSONObject) -> VerboseSegment:
    """Decode JSON object to VerboseSegment with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated VerboseSegment.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    return VerboseSegment(
        text=require_str(obj, "text"),
        start=require_float(obj, "start"),
        end=require_float(obj, "end"),
    )


def require_verbose_segment(obj: JSONValue) -> VerboseSegment:
    """Validate and convert JSONValue to VerboseSegment.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated VerboseSegment.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_verbose_segment(obj)


def encode_verbose_response(response: VerboseResponse) -> JSONObject:
    """Encode VerboseResponse to JSON-compatible dict.

    Args:
        response: The response to encode.

    Returns:
        JSON-compatible dictionary.
    """
    segments: list[JSONValue] = [encode_verbose_segment(s) for s in response["segments"]]
    return {
        "text": response["text"],
        "language": response["language"],
        "segments": segments,
    }


def decode_verbose_response(obj: JSONObject) -> VerboseResponse:
    """Decode JSON object to VerboseResponse with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated VerboseResponse.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    text = require_str(obj, "text")
    language = optional_str(obj, "language")
    segments_raw = require_list(obj, "segments")
    segments: list[VerboseSegment] = []
    for item in segments_raw:
        segments.append(require_verbose_segment(item))
    return VerboseResponse(text=text, language=language, segments=segments)


def require_verbose_response(obj: JSONValue) -> VerboseResponse:
    """Validate and convert JSONValue to VerboseResponse.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated VerboseResponse.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_verbose_response(obj)


# =============================================================================
# Translation Request/Response
# =============================================================================
