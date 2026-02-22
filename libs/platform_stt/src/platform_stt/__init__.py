"""Platform STT - Speech-to-text library with Whisper API, chunking, and language detection.

This library provides:
- OpenAI Whisper API client with transcription and translation support
- Audio chunking with silence detection for large file processing
- Parallel transcription with bounded concurrency
- Segment merging for chunked transcripts
- Language detection using FastText models

Usage:
    from platform_stt import OpenAISttClient, AudioChunker, TranscriptMerger

    # Simple transcription
    client = OpenAISttClient(api_key="...")
    with open("audio.mp3", "rb") as f:
        result = client.transcribe(file=f, language="vi")

    # Translation to English
    with open("audio.mp3", "rb") as f:
        result = client.translate(file=f)
"""

from platform_stt.chunker import AudioChunker
from platform_stt.langid import (
    detect_language,
    ensure_model_path,
    is_language,
    load_langid_model,
)
from platform_stt.merger import TranscriptMerger, merge_segment_text
from platform_stt.parallel import ParallelTranscriber, TranscriptSegmentList
from platform_stt.srt import (
    SrtEntry,
    decode_srt_entry,
    encode_srt_entry,
    format_srt,
    format_srt_entry,
    format_timestamp,
    require_srt_entry,
    segments_to_srt_entries,
    write_srt,
)
from platform_stt.types import (
    WHISPER_SUPPORTED_LANGUAGES,
    AudioChunk,
    BinaryFileProtocol,
    ChunkerConfig,
    LanguageDetectionResult,
    RawVerboseDict,
    SupportsModelDump,
    SupportsToDictRecursive,
    TranscriptSegment,
    TranslationRequest,
    TranslationResponse,
    VerboseResponse,
    VerboseSegment,
    WhisperTask,
    decode_audio_chunk,
    decode_chunker_config,
    decode_language_detection_result,
    decode_transcript_segment,
    decode_translation_request,
    decode_translation_response,
    decode_verbose_response,
    decode_verbose_segment,
    encode_audio_chunk,
    encode_chunker_config,
    encode_language_detection_result,
    encode_transcript_segment,
    encode_translation_request,
    encode_translation_response,
    encode_verbose_response,
    encode_verbose_segment,
    require_audio_chunk,
    require_chunker_config,
    require_language_detection_result,
    require_transcript_segment,
    require_translation_request,
    require_translation_response,
    require_verbose_response,
    require_verbose_segment,
    validate_whisper_language,
    validate_whisper_task,
)
from platform_stt.whisper_client import OpenAISttClient
from platform_stt.whisper_parse import convert_verbose_to_segments, to_verbose_response

__all__ = [
    "WHISPER_SUPPORTED_LANGUAGES",
    "AudioChunk",
    "AudioChunker",
    "BinaryFileProtocol",
    "ChunkerConfig",
    "LanguageDetectionResult",
    "OpenAISttClient",
    "ParallelTranscriber",
    "RawVerboseDict",
    "SrtEntry",
    "SupportsModelDump",
    "SupportsToDictRecursive",
    "TranscriptMerger",
    "TranscriptSegment",
    "TranscriptSegmentList",
    "TranslationRequest",
    "TranslationResponse",
    "VerboseResponse",
    "VerboseSegment",
    "WhisperTask",
    "convert_verbose_to_segments",
    "decode_audio_chunk",
    "decode_chunker_config",
    "decode_language_detection_result",
    "decode_srt_entry",
    "decode_transcript_segment",
    "decode_translation_request",
    "decode_translation_response",
    "decode_verbose_response",
    "decode_verbose_segment",
    "detect_language",
    "encode_audio_chunk",
    "encode_chunker_config",
    "encode_language_detection_result",
    "encode_srt_entry",
    "encode_transcript_segment",
    "encode_translation_request",
    "encode_translation_response",
    "encode_verbose_response",
    "encode_verbose_segment",
    "ensure_model_path",
    "format_srt",
    "format_srt_entry",
    "format_timestamp",
    "is_language",
    "load_langid_model",
    "merge_segment_text",
    "require_audio_chunk",
    "require_chunker_config",
    "require_language_detection_result",
    "require_srt_entry",
    "require_transcript_segment",
    "require_translation_request",
    "require_translation_response",
    "require_verbose_response",
    "require_verbose_segment",
    "segments_to_srt_entries",
    "to_verbose_response",
    "validate_whisper_language",
    "validate_whisper_task",
    "write_srt",
]
