"""Text translation service using pluggable backends.

This module provides the core translation service that routes requests
to configured backends.
"""

from __future__ import annotations

from . import _test_hooks
from .types import TranslationResult, TranslatorConfig


class Translator:
    """Text translator using configurable backends.

    Routes translation requests to the configured backend
    (Anthropic, DeepL, NLLB, etc.).
    """

    __slots__ = ("_backend",)

    def __init__(self, config: TranslatorConfig) -> None:
        """Initialize translator with configuration.

        Args:
            config: Translator configuration specifying backend and credentials.
        """
        self._backend = _test_hooks.backend_factory(config)

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate text from source to target language.

        Args:
            text: Source text to translate.
            source_language: ISO 639-1 source language code (e.g., "vi").
            target_language: ISO 639-1 target language code (e.g., "en").

        Returns:
            TranslationResult with translated text and metadata.

        Raises:
            ValueError: If text is empty.
        """
        if len(text.strip()) == 0:
            raise ValueError("text cannot be empty")

        return self._backend.translate(text, source_language, target_language)

    @property
    def backend_id(self) -> str:
        """Get the backend identifier.

        Returns:
            String identifier for the configured backend.
        """
        return self._backend.backend_id


def translate_text(
    text: str,
    source_language: str,
    target_language: str,
    config: TranslatorConfig,
) -> TranslationResult:
    """Translate text using configured backend.

    Convenience function that creates a translator and runs translation.
    For repeated translations, create a Translator instance to reuse
    the backend connection.

    Args:
        text: Source text to translate.
        source_language: ISO 639-1 source language code.
        target_language: ISO 639-1 target language code.
        config: Translator configuration.

    Returns:
        TranslationResult with translated text and metadata.

    Raises:
        ValueError: If text is empty.
    """
    translator = create_translator(config)
    return translator.translate(text, source_language, target_language)


def create_translator(config: TranslatorConfig) -> Translator:
    """Create a translator instance.

    Args:
        config: Translator configuration.

    Returns:
        Configured Translator instance.
    """
    return Translator(config)


__all__ = [
    "Translator",
    "create_translator",
    "translate_text",
]
