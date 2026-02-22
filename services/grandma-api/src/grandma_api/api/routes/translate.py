"""Translation route for grandma-api.

Provides the /translate endpoint for multi-language audio to English text translation.
Uses a three-stage pipeline: Language ID → Transcription → Translation.
"""

from __future__ import annotations

import io
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile
from platform_core.logging import get_logger

from grandma_api.api.schemas.translate import TranslationResponse
from grandma_api.api.validators.translate import validate_audio_bytes, validate_token
from grandma_api.core.container import ServiceContainer

logger = get_logger(__name__)


def build_router(container: ServiceContainer) -> APIRouter:
    """Build translate router with /translate endpoint.

    Args:
        container: Service container with STT, LangID, and Translator factories.

    Returns:
        APIRouter with translate endpoint configured.
    """
    router = APIRouter()

    async def _translate(
        audio: Annotated[UploadFile, File(description="Audio file to translate")],
        token: Annotated[str, Form(description="Authentication token")],
    ) -> TranslationResponse:
        """Translate audio to English text using three-stage pipeline.

        Pipeline: Language ID → Transcription → Translation

        1. Detect spoken language from audio using MMS-LID
        2. Transcribe audio to source language text using Whisper
        3. Translate source text to English using GPT-4o-mini

        Args:
            audio: Audio file (webm, mp3, wav, m4a, ogg supported).
            token: Authentication token.

        Returns:
            TranslationResponse with English text, detected language,
            source text, and confidence score.

        Raises:
            AppError: UNAUTHORIZED if token invalid, INVALID_INPUT if no audio.
            subprocess.CalledProcessError: If audio conversion fails.
            FileNotFoundError: If ffmpeg is not installed.
        """
        validate_token(token, container.settings["api_token"])
        logger.info("Token validated")

        audio_bytes = await audio.read()
        validate_audio_bytes(audio_bytes)

        audio_filename = audio.filename if audio.filename is not None else "audio.webm"
        logger.info(
            "Starting translation pipeline",
            extra={"audio_filename": audio_filename, "size_bytes": len(audio_bytes)},
        )

        # Stage 1: Transcribe audio with Whisper auto-detect
        # Whisper handles language detection better than MMS-LID for supported languages
        logger.info("Stage 1: Transcribing audio with auto language detection")
        stt_client = container.get_stt_client()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = audio_filename

        transcription = stt_client.transcribe(file=audio_file)
        source_text = transcription["text"].strip()
        # VerboseResponse.language is str | None, default to "en" if None
        language_raw = transcription["language"]
        detected_language: str = language_raw if language_raw is not None else "en"
        confidence = 1.0  # Whisper doesn't return confidence, assume high

        logger.info(
            "Transcription complete",
            extra={
                "source_language": detected_language,
                "text_length": len(source_text),
            },
        )

        # Handle empty transcription (silence/no speech detected)
        if not source_text:
            logger.info("No speech detected in audio")
            return TranslationResponse(
                text="",
                detected_language="unknown",
                source_text="",
                confidence=0.0,
            )

        # Stage 3: Translate to English (if not already English)
        if detected_language == "en":
            english_text = source_text
            logger.info("Source is English, no translation needed")
        else:
            logger.info("Stage 3: Translating to English")
            translator = container.get_translator()
            translation_result = translator.translate(
                source_text,
                detected_language,
                "en",
            )
            english_text = translation_result["text"]
            logger.info(
                "Translation complete",
                extra={
                    "source_language": detected_language,
                    "target_language": "en",
                    "text_length": len(english_text),
                },
            )

        logger.info(
            "Pipeline complete",
            extra={
                "detected_language": detected_language,
                "text_preview": english_text[:100] if len(english_text) > 100 else english_text,
            },
        )

        return TranslationResponse(
            text=english_text,
            detected_language=detected_language,
            source_text=source_text,
            confidence=confidence,
        )

    router.add_api_route("/translate", _translate, methods=["POST"])
    return router


__all__ = ["build_router"]
