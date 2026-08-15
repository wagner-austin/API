"""Core types for platform_stt.

The TypedDicts and their encode/decode/require trios live in the private
_types_* modules, grouped by the payload they describe; this module is the
public surface that re-exports them."""

from __future__ import annotations

from platform_stt._types_transcript import (
    AudioChunk,
    TranscriptSegment,
    decode_audio_chunk,
    decode_transcript_segment,
    encode_audio_chunk,
    encode_transcript_segment,
    require_audio_chunk,
    require_transcript_segment,
)
from platform_stt._types_translation import (
    BinaryFileProtocol,
    ChunkerConfig,
    LanguageDetectionResult,
    RawVerboseDict,
    SupportsModelDump,
    SupportsToDictRecursive,
    TranslationRequest,
    TranslationResponse,
    decode_chunker_config,
    decode_language_detection_result,
    decode_translation_request,
    decode_translation_response,
    encode_chunker_config,
    encode_language_detection_result,
    encode_translation_request,
    encode_translation_response,
    require_chunker_config,
    require_language_detection_result,
    require_translation_request,
    require_translation_response,
)
from platform_stt._types_verbose import (
    VerboseResponse,
    VerboseSegment,
    decode_verbose_response,
    decode_verbose_segment,
    encode_verbose_response,
    encode_verbose_segment,
    require_verbose_response,
    require_verbose_segment,
)
from platform_stt._types_whisper import (
    WHISPER_SUPPORTED_LANGUAGES,
    WhisperTask,
    validate_whisper_language,
    validate_whisper_task,
)

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
