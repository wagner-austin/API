"""Tests for platform_stt package exports."""

from __future__ import annotations

import platform_stt


class TestPackageExports:
    """Tests for public API exports from __init__.py."""

    def test_exports_types(self) -> None:
        """Verify type exports are available."""
        from platform_stt import (
            AudioChunk,
            ChunkerConfig,
            LanguageDetectionResult,
            TranscriptSegment,
            TranslationRequest,
            TranslationResponse,
            VerboseResponse,
            VerboseSegment,
        )

        # Verify they are the actual types by checking name
        assert AudioChunk.__name__ == "AudioChunk"
        assert ChunkerConfig.__name__ == "ChunkerConfig"
        assert LanguageDetectionResult.__name__ == "LanguageDetectionResult"
        assert TranscriptSegment.__name__ == "TranscriptSegment"
        assert TranslationRequest.__name__ == "TranslationRequest"
        assert TranslationResponse.__name__ == "TranslationResponse"
        assert VerboseResponse.__name__ == "VerboseResponse"
        assert VerboseSegment.__name__ == "VerboseSegment"

    def test_exports_encode_functions(self) -> None:
        """Verify encode function exports."""
        from platform_stt import (
            encode_audio_chunk,
            encode_chunker_config,
            encode_language_detection_result,
            encode_transcript_segment,
            encode_translation_request,
            encode_translation_response,
            encode_verbose_response,
            encode_verbose_segment,
        )

        assert encode_audio_chunk.__name__ == "encode_audio_chunk"
        assert encode_chunker_config.__name__ == "encode_chunker_config"
        assert encode_language_detection_result.__name__ == "encode_language_detection_result"
        assert encode_transcript_segment.__name__ == "encode_transcript_segment"
        assert encode_translation_request.__name__ == "encode_translation_request"
        assert encode_translation_response.__name__ == "encode_translation_response"
        assert encode_verbose_response.__name__ == "encode_verbose_response"
        assert encode_verbose_segment.__name__ == "encode_verbose_segment"

    def test_exports_decode_functions(self) -> None:
        """Verify decode function exports."""
        from platform_stt import (
            decode_audio_chunk,
            decode_chunker_config,
            decode_language_detection_result,
            decode_transcript_segment,
            decode_translation_request,
            decode_translation_response,
            decode_verbose_response,
            decode_verbose_segment,
        )

        assert decode_audio_chunk.__name__ == "decode_audio_chunk"
        assert decode_chunker_config.__name__ == "decode_chunker_config"
        assert decode_language_detection_result.__name__ == "decode_language_detection_result"
        assert decode_transcript_segment.__name__ == "decode_transcript_segment"
        assert decode_translation_request.__name__ == "decode_translation_request"
        assert decode_translation_response.__name__ == "decode_translation_response"
        assert decode_verbose_response.__name__ == "decode_verbose_response"
        assert decode_verbose_segment.__name__ == "decode_verbose_segment"

    def test_exports_require_functions(self) -> None:
        """Verify require function exports."""
        from platform_stt import (
            require_audio_chunk,
            require_chunker_config,
            require_language_detection_result,
            require_transcript_segment,
            require_translation_request,
            require_translation_response,
            require_verbose_response,
            require_verbose_segment,
        )

        assert require_audio_chunk.__name__ == "require_audio_chunk"
        assert require_chunker_config.__name__ == "require_chunker_config"
        assert require_language_detection_result.__name__ == "require_language_detection_result"
        assert require_transcript_segment.__name__ == "require_transcript_segment"
        assert require_translation_request.__name__ == "require_translation_request"
        assert require_translation_response.__name__ == "require_translation_response"
        assert require_verbose_response.__name__ == "require_verbose_response"
        assert require_verbose_segment.__name__ == "require_verbose_segment"

    def test_exports_validators(self) -> None:
        """Verify validator function exports."""
        from platform_stt import (
            validate_whisper_language,
            validate_whisper_task,
        )

        assert validate_whisper_language.__name__ == "validate_whisper_language"
        assert validate_whisper_task.__name__ == "validate_whisper_task"

    def test_exports_whisper_languages(self) -> None:
        """Verify WHISPER_SUPPORTED_LANGUAGES export."""
        from platform_stt import WHISPER_SUPPORTED_LANGUAGES

        assert "en" in WHISPER_SUPPORTED_LANGUAGES
        assert "vi" in WHISPER_SUPPORTED_LANGUAGES

    def test_exports_client(self) -> None:
        """Verify OpenAISttClient export."""
        from platform_stt import OpenAISttClient

        assert OpenAISttClient.__name__ == "OpenAISttClient"

    def test_exports_chunker(self) -> None:
        """Verify AudioChunker export."""
        from platform_stt import AudioChunker

        assert AudioChunker.__name__ == "AudioChunker"

    def test_exports_parallel(self) -> None:
        """Verify ParallelTranscriber export."""
        from platform_stt import ParallelTranscriber

        assert ParallelTranscriber.__name__ == "ParallelTranscriber"

    def test_exports_merger(self) -> None:
        """Verify merger exports."""
        from platform_stt import TranscriptMerger, merge_segment_text

        assert TranscriptMerger.__name__ == "TranscriptMerger"
        assert merge_segment_text.__name__ == "merge_segment_text"

    def test_exports_langid(self) -> None:
        """Verify language detection exports."""
        from platform_stt import (
            detect_language,
            ensure_model_path,
            is_language,
            load_langid_model,
        )

        assert detect_language.__name__ == "detect_language"
        assert ensure_model_path.__name__ == "ensure_model_path"
        assert is_language.__name__ == "is_language"
        assert load_langid_model.__name__ == "load_langid_model"

    def test_exports_whisper_parse(self) -> None:
        """Verify whisper_parse exports."""
        from platform_stt import (
            convert_verbose_to_segments,
            to_verbose_response,
        )

        assert convert_verbose_to_segments.__name__ == "convert_verbose_to_segments"
        assert to_verbose_response.__name__ == "to_verbose_response"

    def test_all_list_exists_and_has_expected_exports(self) -> None:
        """Verify __all__ is defined with expected exports."""
        all_set = set(platform_stt.__all__)
        # Check key exports are present
        assert "AudioChunk" in all_set
        assert "AudioChunker" in all_set
        assert "OpenAISttClient" in all_set
        assert "ParallelTranscriber" in all_set
        assert "TranscriptMerger" in all_set
        assert "WHISPER_SUPPORTED_LANGUAGES" in all_set
        assert "detect_language" in all_set
        assert "to_verbose_response" in all_set
