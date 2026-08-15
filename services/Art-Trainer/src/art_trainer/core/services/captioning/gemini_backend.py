"""Gemini Vision backend for image captioning.

This module provides image captioning using Google's Gemini Vision API.
Requires google-genai package and a valid API key.

Supported models (as of 2026):
- gemini-2.5-flash: Latest fast model with vision support
- gemini-2.0-flash: Fast model with vision support
- gemini-1.5-pro: High-quality model with vision support
"""

from __future__ import annotations

from pathlib import Path

from art_trainer.core.services.captioning import _test_hooks
from art_trainer.core.services.captioning.backends import (
    CaptionBackendError,
    CaptionBackendType,
)

# Supported image formats for Gemini
SUPPORTED_FORMATS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

# Recommended default model
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiCaptioner:
    """Gemini Vision captioning implementation.

    Uses Google's Gemini Vision API for high-quality image descriptions.
    Best suited for small datasets where caption quality is critical.
    """

    _model_name: str
    _api_key: str
    _client: _test_hooks.GeminiClient | None

    def __init__(self, model_name: str, api_key: str) -> None:
        """Initialize Gemini captioner.

        Args:
            model_name: Gemini model name (e.g., "gemini-2.5-flash").
            api_key: Google AI API key.

        Raises:
            ValueError: If api_key is empty.
        """
        if not api_key:
            raise ValueError("Gemini API key is required")

        self._model_name = model_name
        self._api_key = api_key
        self._client = None

    def _ensure_client(self) -> _test_hooks.GeminiClient:
        """Ensure the Gemini client is initialized.

        Returns:
            Gemini client instance.
        """
        if self._client is not None:
            return self._client

        client: _test_hooks.GeminiClient = _test_hooks.Hooks.gemini_client_factory(
            api_key=self._api_key
        )
        self._client = client
        return client

    def caption(self, image_path: Path, trigger_word: str) -> str:
        """Generate caption for an image using Gemini Vision.

        Args:
            image_path: Path to the image file.
            trigger_word: Trigger word to prepend to caption.

        Returns:
            Generated caption with trigger word prefix.

        Raises:
            FileNotFoundError: If image_path does not exist.
            CaptionBackendError: If caption generation fails.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix = image_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise CaptionBackendError(f"Unsupported image format: {suffix}")

        client = self._ensure_client()

        # Load image bytes
        image_bytes = image_path.read_bytes()

        # Determine MIME type
        mime_type = _get_mime_type(suffix)

        part_factory: _test_hooks.GeminiPartFactory = _test_hooks.Hooks.gemini_part_factory

        image_part: _test_hooks.GeminiPart = part_factory.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        # Build prompt for detailed captioning
        prompt = (
            "Describe this image in detail for training an AI image generation "
            "model. Include the subject, style, colors, composition, lighting, "
            "and any notable details. Be specific and descriptive. "
            "Output only the description, no other text."
        )

        # Generate caption
        response: _test_hooks.GeminiResponse = client.models.generate_content(
            model=self._model_name,
            contents=[prompt, image_part],
        )

        # Extract text from response
        text: str = response.text

        return f"{trigger_word}, {text.strip()}"

    @property
    def backend_type(self) -> CaptionBackendType:
        """Get the backend type identifier.

        Returns:
            Backend type string.
        """
        return "gemini"


def _get_mime_type(suffix: str) -> str:
    """Get MIME type for image suffix.

    Args:
        suffix: File suffix (e.g., ".jpg").

    Returns:
        MIME type string.
    """
    mime_map: dict[str, str] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mime_map.get(suffix, "image/jpeg")


__all__ = [
    "GeminiCaptioner",
]
