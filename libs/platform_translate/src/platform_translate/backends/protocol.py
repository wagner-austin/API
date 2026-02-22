"""Protocol definition for translation backends.

Defines the interface that all translation backends must implement.
"""

from __future__ import annotations

from typing import Protocol

from platform_translate.types import TranslationResult


class TranslationBackendProtocol(Protocol):
    """Protocol for translation backends.

    All translation backends must implement this protocol to be used
    with the translator service.
    """

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate text from source to target language.

        Args:
            text: Source text to translate.
            source_language: ISO 639-1 source language code.
            target_language: ISO 639-1 target language code.

        Returns:
            TranslationResult with translated text and metadata.

        Raises:
            TranslationError: If translation fails.
        """
        ...

    @property
    def backend_id(self) -> str:
        """Get the backend identifier.

        Returns:
            String identifier for this backend (e.g., "anthropic").
        """
        ...


__all__ = [
    "TranslationBackendProtocol",
]
