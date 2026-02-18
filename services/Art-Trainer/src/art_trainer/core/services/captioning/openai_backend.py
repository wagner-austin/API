"""OpenAI Vision backend for image captioning.

This module provides image captioning using OpenAI's Vision API.
Requires openai package and a valid API key.

Supported models (as of 2026):
- gpt-4o: Recommended model with high-quality vision understanding
- gpt-4o-mini: Cost-effective model with vision support
- o1: Advanced reasoning model with vision support
"""

from __future__ import annotations

import base64
from pathlib import Path

from art_trainer.core.services.captioning import _test_hooks
from art_trainer.core.services.captioning.backends import (
    CaptionBackendError,
    CaptionBackendType,
)

# Supported image formats for OpenAI Vision
SUPPORTED_FORMATS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

# Recommended default model
DEFAULT_MODEL = "gpt-4o"


class OpenAICaptioner:
    """OpenAI Vision captioning implementation.

    Uses OpenAI's GPT-4 Vision API for high-quality image descriptions.
    Best suited for small datasets where caption quality is critical.
    """

    _model_name: str
    _api_key: str
    _client: _test_hooks.OpenAIClient | None

    def __init__(self, model_name: str, api_key: str) -> None:
        """Initialize OpenAI captioner.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4-vision-preview", "gpt-4o").
            api_key: OpenAI API key.

        Raises:
            ValueError: If api_key is empty.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self._model_name = model_name
        self._api_key = api_key
        self._client = None

    def _ensure_client(self) -> _test_hooks.OpenAIClient:
        """Ensure the OpenAI client is initialized.

        Returns:
            OpenAI client instance.
        """
        if self._client is not None:
            return self._client

        client: _test_hooks.OpenAIClient

        # Use hook if set, otherwise use real client
        hook_factory = _test_hooks.Hooks.openai_client_factory
        if hook_factory is not None:
            client = hook_factory(api_key=self._api_key)
        else:
            # Dynamic import with Protocol type annotation
            openai_raw = __import__("openai", fromlist=["OpenAI"])
            openai_mod: _test_hooks.OpenAIModule = openai_raw
            openai_cls: _test_hooks.OpenAIClientFactory = openai_mod.OpenAI
            client = openai_cls(api_key=self._api_key)

        self._client = client
        return client

    def caption(self, image_path: Path, trigger_word: str) -> str:
        """Generate caption for an image using OpenAI Vision.

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

        # Load image as base64
        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Determine MIME type
        mime_type = _get_mime_type(suffix)

        # Build message content
        image_url = f"data:{mime_type};base64,{image_b64}"

        prompt_text = (
            "Describe this image in detail for training an AI image generation "
            "model. Include the subject, style, colors, composition, lighting, "
            "and any notable details. Be specific and descriptive. "
            "Output only the description, no other text."
        )

        text_content: _test_hooks.OpenAITextContentDict = {
            "type": "text",
            "text": prompt_text,
        }
        image_content: _test_hooks.OpenAIImageContentDict = {
            "type": "image_url",
            "image_url": {"url": image_url},
        }
        message: _test_hooks.OpenAIMessageDict = {
            "role": "user",
            "content": [text_content, image_content],
        }
        messages: list[_test_hooks.OpenAIMessageDict] = [message]

        # Generate caption
        response: _test_hooks.OpenAICompletionResponse = client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=300,
        )

        # Extract text from response
        choice: _test_hooks.OpenAIChoice = response.choices[0]
        response_message: _test_hooks.OpenAIMessage = choice.message
        content = response_message.content

        if content is None:
            raise CaptionBackendError("OpenAI returned empty response")

        return f"{trigger_word}, {content.strip()}"

    @property
    def backend_type(self) -> CaptionBackendType:
        """Get the backend type identifier.

        Returns:
            Backend type string.
        """
        return "openai"


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
    "OpenAICaptioner",
]
