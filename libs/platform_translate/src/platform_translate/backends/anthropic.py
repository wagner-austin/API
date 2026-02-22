"""Anthropic Claude backend for text translation.

Uses Claude API for high-quality text translation.
"""

from __future__ import annotations

from typing import Final, Protocol

from platform_translate.types import TranslationResult

# Backend identifier
BACKEND_ID: Final[str] = "anthropic"

# System prompt for translation
TRANSLATION_SYSTEM_PROMPT: Final[str] = (
    "You are a professional translator. "
    "Translate the given text accurately and naturally.\n"
    "Rules:\n"
    "- Translate ONLY the text provided\n"
    "- Preserve formatting, punctuation, and tone\n"
    "- Do not add explanations or notes\n"
    "- Output ONLY the translation, nothing else"
)


class MessageContentProtocol(Protocol):
    """Protocol for Anthropic message content block."""

    @property
    def type(self) -> str:
        """Content block type."""
        ...

    @property
    def text(self) -> str:
        """Text content."""
        ...


class MessageProtocol(Protocol):
    """Protocol for Anthropic message response."""

    @property
    def content(self) -> list[MessageContentProtocol]:
        """Message content blocks."""
        ...


class MessagesProtocol(Protocol):
    """Protocol for Anthropic messages API."""

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> MessageProtocol:
        """Create a message completion.

        Args:
            model: Model identifier.
            max_tokens: Maximum tokens in response.
            system: System prompt.
            messages: List of message dictionaries.

        Returns:
            Message response.
        """
        ...


class AnthropicClientProtocol(Protocol):
    """Protocol for Anthropic client."""

    @property
    def messages(self) -> MessagesProtocol:
        """Messages API namespace."""
        ...


class AnthropicBackend:
    """Translation backend using Anthropic Claude API.

    Uses Claude for high-quality text translation with support for
    many language pairs.
    """

    __slots__ = ("_client", "_model")

    def __init__(self, client: AnthropicClientProtocol, model: str) -> None:
        """Initialize Anthropic backend.

        Args:
            client: Anthropic API client.
            model: Model identifier (e.g., "claude-3-haiku-20240307").
        """
        self._client = client
        self._model = model

    @property
    def backend_id(self) -> str:
        """Get the backend identifier.

        Returns:
            "anthropic" backend identifier.
        """
        return BACKEND_ID

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate text using Claude API.

        Args:
            text: Source text to translate.
            source_language: ISO 639-1 source language code.
            target_language: ISO 639-1 target language code.

        Returns:
            TranslationResult with translated text.

        Raises:
            ValueError: If text is empty.
            APIError: If Claude API call fails.
        """
        if len(text.strip()) == 0:
            raise ValueError("text cannot be empty")

        user_prompt = f"Translate from {source_language} to {target_language}:\n\n{text}"

        messages: list[dict[str, str]] = [
            {"role": "user", "content": user_prompt},
        ]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=TRANSLATION_SYSTEM_PROMPT,
            messages=messages,
        )

        # Extract text from response
        translated_text = ""
        for block in response.content:
            if block.type == "text":
                translated_text = block.text
                break

        return TranslationResult(
            text=translated_text.strip(),
            source_language=source_language,
            target_language=target_language,
            backend=BACKEND_ID,
        )


def create_anthropic_backend(api_key: str, model: str) -> AnthropicBackend:
    """Create Anthropic backend with given credentials.

    Args:
        api_key: Anthropic API key.
        model: Model identifier.

    Returns:
        Configured AnthropicBackend instance.
    """
    mod = __import__("anthropic")
    client: AnthropicClientProtocol = mod.Anthropic(api_key=api_key)
    return AnthropicBackend(client=client, model=model)


__all__ = [
    "BACKEND_ID",
    "TRANSLATION_SYSTEM_PROMPT",
    "AnthropicBackend",
    "AnthropicClientProtocol",
    "create_anthropic_backend",
]
