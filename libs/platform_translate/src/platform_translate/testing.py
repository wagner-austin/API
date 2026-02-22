"""Public test utilities for platform_translate consumers.

Provides fake implementations and test helpers for services using platform_translate.

Usage:
    from platform_translate.testing import (
        FakeTranslationBackend,
        FakeAnthropicClient,
        reset_hooks,
    )

    # Set up fakes for testing
    from platform_translate import _test_hooks
    _test_hooks.backend_factory = lambda config: FakeTranslationBackend()

    # Reset to production after test
    reset_hooks()
"""

from __future__ import annotations

from . import _test_hooks
from .backends.anthropic import MessageContentProtocol, MessageProtocol
from .backends.openai import (
    ChoiceProtocol,
    CompletionProtocol,
)
from .backends.openai import (
    MessageProtocol as OpenAIMessageProtocol,
)
from .types import TranslationResult, TranslatorConfig

# =============================================================================
# Fake Message Content
# =============================================================================


class FakeMessageContent:
    """Fake Anthropic message content block."""

    __slots__ = ("text", "type")

    def __init__(self, text: str, content_type: str = "text") -> None:
        """Initialize fake content block.

        Args:
            text: Text content.
            content_type: Content type identifier.
        """
        self.text = text
        self.type = content_type


# =============================================================================
# Fake Message
# =============================================================================


class FakeMessage:
    """Fake Anthropic message response."""

    __slots__ = ("content",)

    def __init__(self, text: str) -> None:
        """Initialize fake message with text response.

        Args:
            text: Text content of the response.
        """
        self.content: list[MessageContentProtocol] = [FakeMessageContent(text)]


# =============================================================================
# Fake Messages API
# =============================================================================


class FakeMessages:
    """Fake Anthropic messages API."""

    __slots__ = ("_response_text", "calls")

    def __init__(self, response_text: str = "Translated text") -> None:
        """Initialize fake messages API.

        Args:
            response_text: Text to return from create().
        """
        self._response_text = response_text
        self.calls: list[dict[str, str | int | list[dict[str, str]]]] = []

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> MessageProtocol:
        """Record call and return fake message.

        Args:
            model: Model identifier.
            max_tokens: Maximum tokens.
            system: System prompt.
            messages: Message list.

        Returns:
            FakeMessage with configured response.
        """
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            }
        )
        return FakeMessage(self._response_text)


# =============================================================================
# Fake Anthropic Client
# =============================================================================


class FakeAnthropicClient:
    """Fake Anthropic API client for testing."""

    __slots__ = ("messages",)

    def __init__(self, response_text: str = "Translated text") -> None:
        """Initialize fake client.

        Args:
            response_text: Text to return from message completions.
        """
        self.messages = FakeMessages(response_text)


# =============================================================================
# Fake Translation Backend
# =============================================================================


class FakeTranslationBackend:
    """Fake translation backend for testing.

    Returns configurable translation results without making API calls.
    """

    __slots__ = ("_backend_id", "_translated_text", "call_count")

    def __init__(
        self,
        translated_text: str = "Translated text",
        backend_id: str = "fake",
    ) -> None:
        """Initialize fake backend.

        Args:
            translated_text: Text to return as translation.
            backend_id: Backend identifier to return.
        """
        self._translated_text = translated_text
        self._backend_id = backend_id
        self.call_count = 0

    @property
    def backend_id(self) -> str:
        """Get backend identifier.

        Returns:
            Configured backend identifier.
        """
        return self._backend_id

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Return configured translation result.

        Args:
            text: Source text (must be non-empty).
            source_language: Source language code.
            target_language: Target language code.

        Returns:
            Configured TranslationResult.

        Raises:
            ValueError: If text is empty.
        """
        if len(text.strip()) == 0:
            raise ValueError("text cannot be empty")
        self.call_count += 1
        return TranslationResult(
            text=self._translated_text,
            source_language=source_language,
            target_language=target_language,
            backend=self._backend_id,
        )


# =============================================================================
# Fake OpenAI Message
# =============================================================================


class FakeOpenAIMessage:
    """Fake OpenAI chat message."""

    __slots__ = ("content",)

    def __init__(self, content: str | None) -> None:
        """Initialize fake message.

        Args:
            content: Text content of the message.
        """
        self.content = content


# =============================================================================
# Fake OpenAI Choice
# =============================================================================


class FakeOpenAIChoice:
    """Fake OpenAI chat completion choice."""

    __slots__ = ("message",)

    def __init__(self, content: str | None) -> None:
        """Initialize fake choice.

        Args:
            content: Text content for the message.
        """
        self.message: OpenAIMessageProtocol = FakeOpenAIMessage(content)


# =============================================================================
# Fake OpenAI Completion
# =============================================================================


class FakeOpenAICompletion:
    """Fake OpenAI chat completion response."""

    __slots__ = ("choices",)

    def __init__(self, content: str | None) -> None:
        """Initialize fake completion.

        Args:
            content: Text content for the response.
        """
        self.choices: list[ChoiceProtocol] = [FakeOpenAIChoice(content)]


# =============================================================================
# Fake OpenAI Completions API
# =============================================================================


class FakeOpenAICompletions:
    """Fake OpenAI chat completions API."""

    __slots__ = ("_response_text", "calls", "last_messages")

    def __init__(self, response_text: str | None = "Translated text") -> None:
        """Initialize fake completions API.

        Args:
            response_text: Text to return from create().
        """
        self._response_text = response_text
        self.calls: list[dict[str, str | int | list[dict[str, str]]]] = []
        self.last_messages: list[dict[str, str]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> CompletionProtocol:
        """Record call and return fake completion.

        Args:
            model: Model identifier.
            messages: Message list.
            max_tokens: Maximum tokens.

        Returns:
            FakeOpenAICompletion with configured response.
        """
        self.last_messages = messages
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
        )
        return FakeOpenAICompletion(self._response_text)


# =============================================================================
# Fake OpenAI Chat
# =============================================================================


class FakeOpenAIChat:
    """Fake OpenAI chat namespace."""

    __slots__ = ("completions",)

    def __init__(self, response_text: str | None = "Translated text") -> None:
        """Initialize fake chat.

        Args:
            response_text: Text to return from completions.
        """
        self.completions = FakeOpenAICompletions(response_text)


# =============================================================================
# Fake OpenAI Client
# =============================================================================


class FakeOpenAIClient:
    """Fake OpenAI API client for testing."""

    __slots__ = ("chat",)

    def __init__(self, response_text: str | None = "Translated text") -> None:
        """Initialize fake client.

        Args:
            response_text: Text to return from chat completions.
        """
        self.chat = FakeOpenAIChat(response_text)


# =============================================================================
# Hook Management
# =============================================================================


def set_production_hooks() -> None:
    """Set all hooks to production implementations."""
    _test_hooks.backend_factory = _test_hooks._default_backend_factory


def reset_hooks() -> None:
    """Reset all hooks to production implementations."""
    set_production_hooks()


def make_fake_backend_factory(
    translated_text: str = "Translated text",
    backend_id: str = "fake",
) -> _test_hooks.BackendFactoryProtocol:
    """Create a fake backend factory.

    Args:
        translated_text: Text to return as translation.
        backend_id: Backend identifier to return.

    Returns:
        Factory function that creates FakeTranslationBackend.
    """

    def factory(config: TranslatorConfig) -> FakeTranslationBackend:
        del config
        return FakeTranslationBackend(
            translated_text=translated_text,
            backend_id=backend_id,
        )

    return factory


__all__ = [
    "FakeAnthropicClient",
    "FakeMessage",
    "FakeMessageContent",
    "FakeMessages",
    "FakeOpenAIChat",
    "FakeOpenAIChoice",
    "FakeOpenAIClient",
    "FakeOpenAICompletion",
    "FakeOpenAICompletions",
    "FakeOpenAIMessage",
    "FakeTranslationBackend",
    "make_fake_backend_factory",
    "reset_hooks",
    "set_production_hooks",
]
