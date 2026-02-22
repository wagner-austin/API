"""OpenAI GPT backend for text translation.

Uses OpenAI Chat Completions API for text translation.
"""

from __future__ import annotations

from typing import Final, Protocol

from platform_translate.types import TranslationResult

# Backend identifier
BACKEND_ID: Final[str] = "openai"

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


class MessageProtocol(Protocol):
    """Protocol for OpenAI chat message."""

    @property
    def content(self) -> str | None:
        """Message content."""
        ...


class ChoiceProtocol(Protocol):
    """Protocol for OpenAI chat completion choice."""

    @property
    def message(self) -> MessageProtocol:
        """The message in this choice."""
        ...


class CompletionProtocol(Protocol):
    """Protocol for OpenAI chat completion response."""

    @property
    def choices(self) -> list[ChoiceProtocol]:
        """List of completion choices."""
        ...


class CompletionsProtocol(Protocol):
    """Protocol for OpenAI chat completions API."""

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> CompletionProtocol:
        """Create a chat completion.

        Args:
            model: Model identifier.
            messages: List of message dictionaries.
            max_tokens: Maximum tokens in response.

        Returns:
            Completion response.
        """
        ...


class ChatProtocol(Protocol):
    """Protocol for OpenAI chat namespace."""

    @property
    def completions(self) -> CompletionsProtocol:
        """Completions API namespace."""
        ...


class OpenAIClientProtocol(Protocol):
    """Protocol for OpenAI client."""

    @property
    def chat(self) -> ChatProtocol:
        """Chat API namespace."""
        ...


class OpenAIBackend:
    """Translation backend using OpenAI GPT API.

    Uses GPT models for text translation with support for
    many language pairs.
    """

    __slots__ = ("_client", "_model")

    def __init__(self, client: OpenAIClientProtocol, model: str) -> None:
        """Initialize OpenAI backend.

        Args:
            client: OpenAI API client.
            model: Model identifier (e.g., "gpt-4o-mini").
        """
        self._client = client
        self._model = model

    @property
    def backend_id(self) -> str:
        """Get the backend identifier.

        Returns:
            "openai" backend identifier.
        """
        return BACKEND_ID

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Translate text using OpenAI API.

        Args:
            text: Source text to translate.
            source_language: ISO 639-1 source language code.
            target_language: ISO 639-1 target language code.

        Returns:
            TranslationResult with translated text.

        Raises:
            ValueError: If text is empty or response is empty.
        """
        if len(text.strip()) == 0:
            raise ValueError("text cannot be empty")

        user_prompt = f"Translate from {source_language} to {target_language}:\n\n{text}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=4096,
        )

        # Extract text from response
        translated_text = ""
        if len(response.choices) > 0:
            content = response.choices[0].message.content
            if content is not None:
                translated_text = content

        return TranslationResult(
            text=translated_text.strip(),
            source_language=source_language,
            target_language=target_language,
            backend=BACKEND_ID,
        )


def create_openai_backend(api_key: str, model: str) -> OpenAIBackend:
    """Create OpenAI backend with given credentials.

    Args:
        api_key: OpenAI API key.
        model: Model identifier.

    Returns:
        Configured OpenAIBackend instance.
    """
    mod = __import__("openai")
    client: OpenAIClientProtocol = mod.OpenAI(api_key=api_key)
    return OpenAIBackend(client=client, model=model)


__all__ = [
    "BACKEND_ID",
    "TRANSLATION_SYSTEM_PROMPT",
    "OpenAIBackend",
    "OpenAIClientProtocol",
    "create_openai_backend",
]
