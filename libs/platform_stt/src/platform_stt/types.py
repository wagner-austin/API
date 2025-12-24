"""Type definitions for platform_stt library.

All TypedDicts include encode/decode functions with require_* validation.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_str,
    require_float,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

# =============================================================================
# Whisper Task Types
# =============================================================================

WhisperTask = Literal["transcribe", "translate"]

# Supported languages for Whisper (ISO 639-1 codes)
# Full list at: https://platform.openai.com/docs/guides/speech-to-text
WHISPER_SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {
        "af",
        "ar",
        "hy",
        "az",
        "be",
        "bs",
        "bg",
        "ca",
        "zh",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "gl",
        "de",
        "el",
        "he",
        "hi",
        "hu",
        "is",
        "id",
        "it",
        "ja",
        "kn",
        "kk",
        "ko",
        "lv",
        "lt",
        "mk",
        "ms",
        "mr",
        "mi",
        "ne",
        "no",
        "fa",
        "pl",
        "pt",
        "ro",
        "ru",
        "sr",
        "sk",
        "sl",
        "es",
        "sw",
        "sv",
        "tl",
        "ta",
        "th",
        "tr",
        "uk",
        "ur",
        "vi",
        "cy",
    }
)


def validate_whisper_language(lang: str) -> str:
    """Validate that a language code is supported by Whisper.

    Args:
        lang: ISO 639-1 language code.

    Returns:
        The validated language code.

    Raises:
        ValueError: If the language is not supported.
    """
    if lang not in WHISPER_SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported Whisper language: {lang}")
    return lang


def validate_whisper_task(task: str) -> WhisperTask:
    """Validate that a task is a valid Whisper task.

    Args:
        task: Task string to validate.

    Returns:
        The validated task as WhisperTask literal.

    Raises:
        ValueError: If the task is not valid.
    """
    if task not in ("transcribe", "translate"):
        raise ValueError(f"Invalid Whisper task: {task}")
    if task == "transcribe":
        return "transcribe"
    return "translate"


# =============================================================================
# Transcript Segment
# =============================================================================


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


class VerboseSegment(TypedDict):
    """A segment from OpenAI Whisper verbose response."""

    text: str
    start: float
    end: float


class VerboseResponse(TypedDict):
    """OpenAI Whisper verbose_json response format."""

    text: str
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
    segments_raw = require_list(obj, "segments")
    segments: list[VerboseSegment] = []
    for item in segments_raw:
        segments.append(require_verbose_segment(item))
    return VerboseResponse(text=text, segments=segments)


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


__all__ = [
    "WHISPER_SUPPORTED_LANGUAGES",
    "AudioChunk",
    "BinaryFileProtocol",
    "ChunkerConfig",
    "LanguageDetectionResult",
    "RawVerboseDict",
    "SupportsModelDump",
    "SupportsToDictRecursive",
    "TranscriptSegment",
    "TranslationRequest",
    "TranslationResponse",
    "VerboseResponse",
    "VerboseSegment",
    "WhisperTask",
    "decode_audio_chunk",
    "decode_chunker_config",
    "decode_language_detection_result",
    "decode_transcript_segment",
    "decode_translation_request",
    "decode_translation_response",
    "decode_verbose_response",
    "decode_verbose_segment",
    "encode_audio_chunk",
    "encode_chunker_config",
    "encode_language_detection_result",
    "encode_transcript_segment",
    "encode_translation_request",
    "encode_translation_response",
    "encode_verbose_response",
    "encode_verbose_segment",
    "require_audio_chunk",
    "require_chunker_config",
    "require_language_detection_result",
    "require_transcript_segment",
    "require_translation_request",
    "require_translation_response",
    "require_verbose_response",
    "require_verbose_segment",
    "validate_whisper_language",
    "validate_whisper_task",
]
