"""Translation route for grandma-api.

Provides the /translate endpoint for multi-language audio to English text translation.
Supports 57 input languages with automatic language detection via OpenAI Whisper.
Uses ServiceContainer for dependency injection and proper validation.
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
        container: Service container with STT client factory.

    Returns:
        APIRouter with translate endpoint configured.
    """
    router = APIRouter()

    async def _translate(
        audio: Annotated[UploadFile, File(description="Audio file to translate")],
        token: Annotated[str, Form(description="Authentication token")],
    ) -> TranslationResponse:
        """Translate audio to English text.

        Supports 57 input languages with automatic language detection.
        Output is always English (Whisper API limitation).

        Args:
            audio: Audio file (webm, mp3, wav, m4a, ogg supported).
            token: Authentication token.

        Returns:
            TranslationResponse with English text.

        Raises:
            AppError: UNAUTHORIZED if token invalid, INVALID_INPUT if no audio.
        """
        # Validate token
        validate_token(token, container.settings["api_token"])

        # Read and validate audio
        audio_bytes = await audio.read()
        validate_audio_bytes(audio_bytes)

        audio_filename = audio.filename if audio.filename is not None else "audio.webm"
        logger.info(
            "Translating audio",
            extra={"audio_filename": audio_filename, "size_bytes": len(audio_bytes)},
        )

        # Get STT client from container and translate
        client = container.get_stt_client()

        # Create file-like object for the STT client
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = audio_filename

        result = client.translate(file=audio_file)

        text = result["text"]
        language = result["language"]
        text_preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(
            "Translation complete",
            extra={
                "detected_language": language,
                "text_length": len(text),
                "text_preview": text_preview,
            },
        )

        return TranslationResponse(text=text)

    router.add_api_route("/translate", _translate, methods=["POST"])
    return router


__all__ = ["build_router"]
