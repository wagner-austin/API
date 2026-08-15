"""types: TranscriptSegment and related definitions."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict


class TranscriptSegment(TypedDict):
    """A single segment of transcribed audio with timing information."""

    text: str
    start: float
    duration: float


def encode_transcript_segment(segment: TranscriptSegment) -> JSONObject:
    """Encode TranscriptSegment to JSON-compatible dict.

    Args:
        segment: The segment to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "text": segment["text"],
        "start": segment["start"],
        "duration": segment["duration"],
    }


def decode_transcript_segment(obj: JSONObject) -> TranscriptSegment:
    """Decode JSON object to TranscriptSegment with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TranscriptSegment.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    return TranscriptSegment(
        text=require_str(obj, "text"),
        start=require_float(obj, "start"),
        duration=require_float(obj, "duration"),
    )


def require_transcript_segment(obj: JSONValue) -> TranscriptSegment:
    """Validate and convert JSONValue to TranscriptSegment.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated TranscriptSegment.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_transcript_segment(obj)


# =============================================================================
# Audio Chunk
# =============================================================================


class AudioChunk(TypedDict):
    """Represents a physical audio file chunk and its time window in the source."""

    path: str
    start_seconds: float
    duration_seconds: float
    size_bytes: int


def encode_audio_chunk(chunk: AudioChunk) -> JSONObject:
    """Encode AudioChunk to JSON-compatible dict.

    Args:
        chunk: The chunk to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "path": chunk["path"],
        "start_seconds": chunk["start_seconds"],
        "duration_seconds": chunk["duration_seconds"],
        "size_bytes": chunk["size_bytes"],
    }


def decode_audio_chunk(obj: JSONObject) -> AudioChunk:
    """Decode JSON object to AudioChunk with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated AudioChunk.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    return AudioChunk(
        path=require_str(obj, "path"),
        start_seconds=require_float(obj, "start_seconds"),
        duration_seconds=require_float(obj, "duration_seconds"),
        size_bytes=require_int(obj, "size_bytes"),
    )


def require_audio_chunk(obj: JSONValue) -> AudioChunk:
    """Validate and convert JSONValue to AudioChunk.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated AudioChunk.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_audio_chunk(obj)


# =============================================================================
# Verbose Response (OpenAI Whisper API response format)
# =============================================================================
