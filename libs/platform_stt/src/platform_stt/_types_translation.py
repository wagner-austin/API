"""types: TranslationRequest and related definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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

from platform_stt._types_transcript import (
    TranscriptSegment,
    encode_transcript_segment,
    require_transcript_segment,
)
from platform_stt._types_whisper import WhisperTask, validate_whisper_task


class TranslationRequest(TypedDict):
    """Request for audio translation."""

    source_language: str | None  # None for auto-detect
    target_language: str  # Usually "en" for translation
    task: WhisperTask


class TranslationResponse(TypedDict):
    """Response from audio translation."""

    text: str
    detected_language: str | None
    segments: list[TranscriptSegment]


def encode_translation_request(request: TranslationRequest) -> JSONObject:
    """Encode TranslationRequest to JSON-compatible dict.

    Args:
        request: The request to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "source_language": request["source_language"],
        "target_language": request["target_language"],
        "task": request["task"],
    }


def decode_translation_request(obj: JSONObject) -> TranslationRequest:
    """Decode JSON object to TranslationRequest with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated TranslationRequest.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    source_language = optional_str(obj, "source_language")
    target_language = require_str(obj, "target_language")
    task_raw = require_str(obj, "task")
    task = validate_whisper_task(task_raw)
    return TranslationRequest(
        source_language=source_language,
        target_language=target_language,
        task=task,
    )


def require_translation_request(obj: JSONValue) -> TranslationRequest:
    """Validate and convert JSONValue to TranslationRequest.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated TranslationRequest.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_translation_request(obj)


def encode_translation_response(response: TranslationResponse) -> JSONObject:
    """Encode TranslationResponse to JSON-compatible dict.

    Args:
        response: The response to encode.

    Returns:
        JSON-compatible dictionary.
    """
    segments: list[JSONValue] = [encode_transcript_segment(s) for s in response["segments"]]
    return {
        "text": response["text"],
        "detected_language": response["detected_language"],
        "segments": segments,
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
    detected_language = optional_str(obj, "detected_language")
    segments_raw = require_list(obj, "segments")
    segments: list[TranscriptSegment] = []
    for item in segments_raw:
        segments.append(require_transcript_segment(item))
    return TranslationResponse(
        text=text,
        detected_language=detected_language,
        segments=segments,
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


# =============================================================================
# Language Detection Result
# =============================================================================


class LanguageDetectionResult(TypedDict):
    """Result of language detection on audio or text."""

    language: str
    confidence: float
    script: str | None


def encode_language_detection_result(result: LanguageDetectionResult) -> JSONObject:
    """Encode LanguageDetectionResult to JSON-compatible dict.

    Args:
        result: The result to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "language": result["language"],
        "confidence": result["confidence"],
        "script": result["script"],
    }


def decode_language_detection_result(obj: JSONObject) -> LanguageDetectionResult:
    """Decode JSON object to LanguageDetectionResult with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated LanguageDetectionResult.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    return LanguageDetectionResult(
        language=require_str(obj, "language"),
        confidence=require_float(obj, "confidence"),
        script=optional_str(obj, "script"),
    )


def require_language_detection_result(obj: JSONValue) -> LanguageDetectionResult:
    """Validate and convert JSONValue to LanguageDetectionResult.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated LanguageDetectionResult.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_language_detection_result(obj)


# =============================================================================
# Chunker Configuration
# =============================================================================


class ChunkerConfig(TypedDict):
    """Configuration for audio chunking."""

    target_chunk_mb: float
    max_chunk_duration_seconds: float
    silence_threshold_db: float
    silence_duration_seconds: float


def encode_chunker_config(config: ChunkerConfig) -> JSONObject:
    """Encode ChunkerConfig to JSON-compatible dict.

    Args:
        config: The config to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "target_chunk_mb": config["target_chunk_mb"],
        "max_chunk_duration_seconds": config["max_chunk_duration_seconds"],
        "silence_threshold_db": config["silence_threshold_db"],
        "silence_duration_seconds": config["silence_duration_seconds"],
    }


def decode_chunker_config(obj: JSONObject) -> ChunkerConfig:
    """Decode JSON object to ChunkerConfig with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated ChunkerConfig.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    target = require_float(obj, "target_chunk_mb")
    max_dur = require_float(obj, "max_chunk_duration_seconds")
    threshold = require_float(obj, "silence_threshold_db")
    silence_dur = require_float(obj, "silence_duration_seconds")

    # Validate reasonable ranges
    if target < 1.0:
        raise JSONTypeError("target_chunk_mb must be >= 1.0")
    if max_dur < 1.0:
        raise JSONTypeError("max_chunk_duration_seconds must be >= 1.0")
    if silence_dur < 0.1:
        raise JSONTypeError("silence_duration_seconds must be >= 0.1")

    return ChunkerConfig(
        target_chunk_mb=target,
        max_chunk_duration_seconds=max_dur,
        silence_threshold_db=threshold,
        silence_duration_seconds=silence_dur,
    )


def require_chunker_config(obj: JSONValue) -> ChunkerConfig:
    """Validate and convert JSONValue to ChunkerConfig.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated ChunkerConfig.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_chunker_config(obj)


# =============================================================================
# Protocols for External Dependencies
# =============================================================================


@runtime_checkable
class BinaryFileProtocol(Protocol):
    """Protocol for binary file-like objects."""

    def read(self, size: int = -1) -> bytes:
        """Read bytes from the file."""
        ...

    def close(self) -> None:
        """Close the file."""
        ...


@runtime_checkable
class SupportsToDictRecursive(Protocol):
    """Protocol for objects that support to_dict_recursive method."""

    def to_dict_recursive(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
        """Convert to dictionary recursively."""
        ...


@runtime_checkable
class SupportsModelDump(Protocol):
    """Protocol for objects that support model_dump method (Pydantic v2)."""

    def model_dump(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]:
        """Dump model to dictionary."""
        ...


# Type alias for raw verbose response from SDK
RawVerboseDict = dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]
