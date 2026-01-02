"""Tests for platform_stt.types module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_stt.types import (
    WHISPER_SUPPORTED_LANGUAGES,
    AudioChunk,
    ChunkerConfig,
    LanguageDetectionResult,
    TranscriptSegment,
    TranslationRequest,
    TranslationResponse,
    VerboseResponse,
    VerboseSegment,
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


class TestWhisperTask:
    """Tests for WhisperTask validation."""

    def test_validate_whisper_task_transcribe(self) -> None:
        """Validate 'transcribe' task."""
        result = validate_whisper_task("transcribe")
        assert result == "transcribe"

    def test_validate_whisper_task_translate(self) -> None:
        """Validate 'translate' task."""
        result = validate_whisper_task("translate")
        assert result == "translate"

    def test_validate_whisper_task_invalid(self) -> None:
        """Reject invalid task."""
        with pytest.raises(ValueError, match="Invalid Whisper task"):
            validate_whisper_task("invalid")


class TestWhisperLanguage:
    """Tests for Whisper language validation."""

    def test_validate_whisper_language_vi(self) -> None:
        """Validate Vietnamese."""
        result = validate_whisper_language("vi")
        assert result == "vi"

    def test_validate_whisper_language_en(self) -> None:
        """Validate English."""
        result = validate_whisper_language("en")
        assert result == "en"

    def test_validate_whisper_language_invalid(self) -> None:
        """Reject invalid language."""
        with pytest.raises(ValueError, match="Unsupported Whisper language"):
            validate_whisper_language("xx")

    def test_whisper_supported_languages_contains_expected(self) -> None:
        """Verify WHISPER_SUPPORTED_LANGUAGES contains expected languages."""
        assert "vi" in WHISPER_SUPPORTED_LANGUAGES
        assert "en" in WHISPER_SUPPORTED_LANGUAGES


class TestTranscriptSegment:
    """Tests for TranscriptSegment encode/decode."""

    def test_encode_transcript_segment(self) -> None:
        """Encode TranscriptSegment to dict."""
        segment = TranscriptSegment(text="hello", start=1.0, duration=2.0)
        result = encode_transcript_segment(segment)
        assert result == {"text": "hello", "start": 1.0, "duration": 2.0}

    def test_decode_transcript_segment(self) -> None:
        """Decode dict to TranscriptSegment."""
        obj: JSONObject = {"text": "hello", "start": 1.0, "duration": 2.0}
        result = decode_transcript_segment(obj)
        assert result["text"] == "hello"
        assert result["start"] == 1.0
        assert result["duration"] == 2.0

    def test_decode_transcript_segment_missing_text(self) -> None:
        """Reject missing text field."""
        with pytest.raises(JSONTypeError, match="Missing required field"):
            decode_transcript_segment({"start": 1.0, "duration": 2.0})

    def test_require_transcript_segment(self) -> None:
        """Validate and convert JSONValue."""
        obj: JSONObject = {"text": "test", "start": 0.0, "duration": 1.0}
        result = require_transcript_segment(obj)
        assert result["text"] == "test"

    def test_require_transcript_segment_not_dict(self) -> None:
        """Reject non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_transcript_segment("not a dict")


class TestAudioChunk:
    """Tests for AudioChunk encode/decode."""

    def test_encode_audio_chunk(self) -> None:
        """Encode AudioChunk to dict."""
        chunk = AudioChunk(
            path="/tmp/chunk.mp3",
            start_seconds=0.0,
            duration_seconds=60.0,
            size_bytes=1024,
        )
        result = encode_audio_chunk(chunk)
        assert result["path"] == "/tmp/chunk.mp3"
        assert result["size_bytes"] == 1024

    def test_decode_audio_chunk(self) -> None:
        """Decode dict to AudioChunk."""
        obj: JSONObject = {
            "path": "/tmp/chunk.mp3",
            "start_seconds": 0.0,
            "duration_seconds": 60.0,
            "size_bytes": 1024,
        }
        result = decode_audio_chunk(obj)
        assert result["path"] == "/tmp/chunk.mp3"
        assert result["size_bytes"] == 1024

    def test_require_audio_chunk(self) -> None:
        """Validate and convert JSONValue."""
        obj: JSONObject = {
            "path": "/tmp/test.mp3",
            "start_seconds": 0.0,
            "duration_seconds": 30.0,
            "size_bytes": 512,
        }
        result = require_audio_chunk(obj)
        assert result["path"] == "/tmp/test.mp3"

    def test_require_audio_chunk_not_dict(self) -> None:
        """Reject non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_audio_chunk([1, 2, 3])


class TestVerboseSegment:
    """Tests for VerboseSegment encode/decode."""

    def test_encode_verbose_segment(self) -> None:
        """Encode VerboseSegment to dict."""
        segment = VerboseSegment(text="hello", start=0.0, end=1.0)
        result = encode_verbose_segment(segment)
        assert result == {"text": "hello", "start": 0.0, "end": 1.0}

    def test_decode_verbose_segment(self) -> None:
        """Decode dict to VerboseSegment."""
        obj: JSONObject = {"text": "hello", "start": 0.0, "end": 1.0}
        result = decode_verbose_segment(obj)
        assert result["text"] == "hello"
        assert result["end"] == 1.0

    def test_require_verbose_segment(self) -> None:
        """Validate and convert JSONValue."""
        obj: JSONObject = {"text": "test", "start": 0.0, "end": 2.0}
        result = require_verbose_segment(obj)
        assert result["end"] == 2.0

    def test_require_verbose_segment_not_dict(self) -> None:
        """Reject non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_verbose_segment(123)


class TestVerboseResponse:
    """Tests for VerboseResponse encode/decode."""

    def test_encode_verbose_response(self) -> None:
        """Encode VerboseResponse to dict."""
        response = VerboseResponse(
            text="hello world",
            language="en",
            segments=[VerboseSegment(text="hello", start=0.0, end=0.5)],
        )
        result = encode_verbose_response(response)
        assert result["text"] == "hello world"
        assert result["language"] == "en"
        # Check segments is correctly encoded
        assert result["segments"] == [{"text": "hello", "start": 0.0, "end": 0.5}]

    def test_decode_verbose_response(self) -> None:
        """Decode dict to VerboseResponse."""
        obj: JSONObject = {
            "text": "hello world",
            "language": "en",
            "segments": [{"text": "hello", "start": 0.0, "end": 0.5}],
        }
        result = decode_verbose_response(obj)
        assert result["text"] == "hello world"
        assert result["language"] == "en"
        assert len(result["segments"]) == 1

    def test_decode_verbose_response_no_language(self) -> None:
        """Decode dict to VerboseResponse when language is missing."""
        obj: JSONObject = {
            "text": "hello world",
            "segments": [{"text": "hello", "start": 0.0, "end": 0.5}],
        }
        result = decode_verbose_response(obj)
        assert result["text"] == "hello world"
        assert result["language"] is None
        assert len(result["segments"]) == 1

    def test_require_verbose_response(self) -> None:
        """Validate and convert JSONValue."""
        obj: JSONObject = {"text": "test", "language": "vi", "segments": []}
        result = require_verbose_response(obj)
        assert result["text"] == "test"
        assert result["language"] == "vi"

    def test_require_verbose_response_not_dict(self) -> None:
        """Reject non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_verbose_response(None)


class TestTranslationRequest:
    """Tests for TranslationRequest encode/decode."""

    def test_encode_translation_request(self) -> None:
        """Encode TranslationRequest to dict."""
        request = TranslationRequest(
            source_language="vi",
            target_language="en",
            task="translate",
        )
        result = encode_translation_request(request)
        assert result["source_language"] == "vi"
        assert result["task"] == "translate"

    def test_decode_translation_request(self) -> None:
        """Decode dict to TranslationRequest."""
        obj: JSONObject = {
            "source_language": "vi",
            "target_language": "en",
            "task": "translate",
        }
        result = decode_translation_request(obj)
        assert result["source_language"] == "vi"
        assert result["task"] == "translate"

    def test_decode_translation_request_none_source(self) -> None:
        """Decode with None source language."""
        obj: JSONObject = {
            "source_language": None,
            "target_language": "en",
            "task": "transcribe",
        }
        result = decode_translation_request(obj)
        assert result["source_language"] is None

    def test_require_translation_request(self) -> None:
        """Validate and convert JSONValue."""
        obj: JSONObject = {"target_language": "en", "task": "transcribe"}
        result = require_translation_request(obj)
        assert result["target_language"] == "en"

    def test_require_translation_request_not_dict(self) -> None:
        """Reject non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_translation_request("string")


class TestTranslationResponse:
    """Tests for TranslationResponse encode/decode."""

    def test_encode_translation_response(self) -> None:
        """Encode TranslationResponse to dict."""
        response = TranslationResponse(
            text="Hello",
            detected_language="vi",
            segments=[TranscriptSegment(text="Hello", start=0.0, duration=1.0)],
        )
        result = encode_translation_response(response)
        assert result["text"] == "Hello"
        assert result["detected_language"] == "vi"

    def test_decode_translation_response(self) -> None:
        """Decode dict to TranslationResponse."""
        obj: JSONObject = {
            "text": "Hello",
            "detected_language": "vi",
            "segments": [{"text": "Hello", "start": 0.0, "duration": 1.0}],
        }
        result = decode_translation_response(obj)
        assert result["text"] == "Hello"
        assert len(result["segments"]) == 1

    def test_require_translation_response(self) -> None:
        """Validate and convert JSONValue."""
        obj: JSONObject = {"text": "test", "segments": []}
        result = require_translation_response(obj)
        assert result["text"] == "test"

    def test_require_translation_response_not_dict(self) -> None:
        """Reject non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_translation_response(False)


class TestLanguageDetectionResult:
    """Tests for LanguageDetectionResult encode/decode."""

    def test_encode_language_detection_result(self) -> None:
        """Encode LanguageDetectionResult to dict."""
        result_obj = LanguageDetectionResult(
            language="vi",
            confidence=0.95,
            script="Latn",
        )
        result = encode_language_detection_result(result_obj)
        assert result["language"] == "vi"
        assert result["script"] == "Latn"

    def test_decode_language_detection_result(self) -> None:
        """Decode dict to LanguageDetectionResult."""
        obj: JSONObject = {"language": "vi", "confidence": 0.95, "script": None}
        result = decode_language_detection_result(obj)
        assert result["language"] == "vi"
        assert result["script"] is None

    def test_require_language_detection_result(self) -> None:
        """Validate and convert JSONValue."""
        obj: JSONObject = {"language": "en", "confidence": 0.99, "script": "Latn"}
        result = require_language_detection_result(obj)
        assert result["language"] == "en"

    def test_require_language_detection_result_not_dict(self) -> None:
        """Reject non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_language_detection_result(42)


class TestChunkerConfig:
    """Tests for ChunkerConfig encode/decode."""

    def test_encode_chunker_config(self) -> None:
        """Encode ChunkerConfig to dict."""
        config = ChunkerConfig(
            target_chunk_mb=20.0,
            max_chunk_duration_seconds=600.0,
            silence_threshold_db=-40.0,
            silence_duration_seconds=0.5,
        )
        result = encode_chunker_config(config)
        assert result["target_chunk_mb"] == 20.0

    def test_decode_chunker_config(self) -> None:
        """Decode dict to ChunkerConfig."""
        obj: JSONObject = {
            "target_chunk_mb": 20.0,
            "max_chunk_duration_seconds": 600.0,
            "silence_threshold_db": -40.0,
            "silence_duration_seconds": 0.5,
        }
        result = decode_chunker_config(obj)
        assert result["target_chunk_mb"] == 20.0

    def test_decode_chunker_config_invalid_target_chunk(self) -> None:
        """Reject target_chunk_mb < 1.0."""
        obj: JSONObject = {
            "target_chunk_mb": 0.5,
            "max_chunk_duration_seconds": 600.0,
            "silence_threshold_db": -40.0,
            "silence_duration_seconds": 0.5,
        }
        with pytest.raises(JSONTypeError, match=r"target_chunk_mb must be >= 1\.0"):
            decode_chunker_config(obj)

    def test_decode_chunker_config_invalid_max_duration(self) -> None:
        """Reject max_chunk_duration_seconds < 1.0."""
        obj: JSONObject = {
            "target_chunk_mb": 20.0,
            "max_chunk_duration_seconds": 0.5,
            "silence_threshold_db": -40.0,
            "silence_duration_seconds": 0.5,
        }
        with pytest.raises(JSONTypeError, match=r"max_chunk_duration_seconds must be >= 1\.0"):
            decode_chunker_config(obj)

    def test_decode_chunker_config_invalid_silence_duration(self) -> None:
        """Reject silence_duration_seconds < 0.1."""
        obj: JSONObject = {
            "target_chunk_mb": 20.0,
            "max_chunk_duration_seconds": 600.0,
            "silence_threshold_db": -40.0,
            "silence_duration_seconds": 0.05,
        }
        with pytest.raises(JSONTypeError, match=r"silence_duration_seconds must be >= 0\.1"):
            decode_chunker_config(obj)

    def test_require_chunker_config(self) -> None:
        """Validate and convert JSONValue."""
        obj: JSONObject = {
            "target_chunk_mb": 10.0,
            "max_chunk_duration_seconds": 300.0,
            "silence_threshold_db": -35.0,
            "silence_duration_seconds": 0.3,
        }
        result = require_chunker_config(obj)
        assert result["target_chunk_mb"] == 10.0

    def test_require_chunker_config_not_dict(self) -> None:
        """Reject non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_chunker_config([])
