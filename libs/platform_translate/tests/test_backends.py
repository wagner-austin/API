"""Tests for platform_translate.backends module."""

from __future__ import annotations

import pytest

from platform_translate.backends.anthropic import (
    BACKEND_ID as ANTHROPIC_BACKEND_ID,
)
from platform_translate.backends.anthropic import (
    TRANSLATION_SYSTEM_PROMPT as ANTHROPIC_SYSTEM_PROMPT,
)
from platform_translate.backends.anthropic import (
    AnthropicBackend,
    MessageContentProtocol,
    MessageProtocol,
)
from platform_translate.backends.openai import (
    BACKEND_ID as OPENAI_BACKEND_ID,
)
from platform_translate.backends.openai import (
    TRANSLATION_SYSTEM_PROMPT as OPENAI_SYSTEM_PROMPT,
)
from platform_translate.backends.openai import (
    ChoiceProtocol,
    CompletionProtocol,
    OpenAIBackend,
)
from platform_translate.backends.protocol import TranslationBackendProtocol
from platform_translate.testing import (
    FakeAnthropicClient,
    FakeMessageContent,
    FakeOpenAIClient,
)


class FakeMessageEmpty:
    """Fake message with empty content list."""

    __slots__ = ("content",)

    def __init__(self) -> None:
        """Initialize with empty content."""
        self.content: list[MessageContentProtocol] = []


class FakeMessageMixed:
    """Fake message with mixed content types."""

    __slots__ = ("content",)

    def __init__(self) -> None:
        """Initialize with mixed content (non-text first, then text)."""
        self.content: list[MessageContentProtocol] = [
            FakeMessageContent("ignored", "tool_use"),
            FakeMessageContent("Found text", "text"),
        ]


class FakeMessagesEmpty:
    """Fake messages API that returns empty content."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        """Initialize with empty calls list."""
        self.calls: list[dict[str, str | int | list[dict[str, str]]]] = []

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> MessageProtocol:
        """Return message with empty content."""
        self.calls.append(
            {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
        )
        return FakeMessageEmpty()


class FakeMessagesMixed:
    """Fake messages API that returns mixed content types."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        """Initialize with empty calls list."""
        self.calls: list[dict[str, str | int | list[dict[str, str]]]] = []

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> MessageProtocol:
        """Return message with mixed content types."""
        self.calls.append(
            {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages}
        )
        return FakeMessageMixed()


class FakeClientEmpty:
    """Fake client that returns empty content."""

    __slots__ = ("messages",)

    def __init__(self) -> None:
        """Initialize with empty messages."""
        self.messages = FakeMessagesEmpty()


class FakeClientMixed:
    """Fake client that returns mixed content types."""

    __slots__ = ("messages",)

    def __init__(self) -> None:
        """Initialize with mixed messages."""
        self.messages = FakeMessagesMixed()


class TestAnthropicBackend:
    """Tests for AnthropicBackend class."""

    def test_backend_id(self) -> None:
        """Backend ID is 'anthropic'."""
        client = FakeAnthropicClient()
        backend = AnthropicBackend(client=client, model="test")
        assert backend.backend_id == "anthropic"
        assert ANTHROPIC_BACKEND_ID == "anthropic"

    def test_translate_returns_result(self) -> None:
        """Translate returns TranslationResult."""
        client = FakeAnthropicClient(response_text="Hola")
        backend = AnthropicBackend(client=client, model="claude-test")
        result = backend.translate("Hello", "en", "es")
        assert result["text"] == "Hola"
        assert result["source_language"] == "en"
        assert result["target_language"] == "es"
        assert result["backend"] == "anthropic"

    def test_translate_calls_api_with_correct_params(self) -> None:
        """Translate calls API with correct parameters."""
        client = FakeAnthropicClient()
        backend = AnthropicBackend(client=client, model="claude-3-haiku")
        backend.translate("Test", "en", "fr")

        assert len(client.messages.calls) == 1
        call = client.messages.calls[0]
        assert call["model"] == "claude-3-haiku"
        assert call["max_tokens"] == 4096
        assert call["system"] == ANTHROPIC_SYSTEM_PROMPT

    def test_translate_empty_raises(self) -> None:
        """Translate raises ValueError for empty text."""
        client = FakeAnthropicClient()
        backend = AnthropicBackend(client=client, model="test")
        with pytest.raises(ValueError, match="cannot be empty"):
            backend.translate("", "en", "es")

    def test_translate_whitespace_raises(self) -> None:
        """Translate raises ValueError for whitespace-only text."""
        client = FakeAnthropicClient()
        backend = AnthropicBackend(client=client, model="test")
        with pytest.raises(ValueError, match="cannot be empty"):
            backend.translate("   ", "en", "es")

    def test_translate_strips_result(self) -> None:
        """Translate strips whitespace from result."""
        client = FakeAnthropicClient(response_text="  Hello  ")
        backend = AnthropicBackend(client=client, model="test")
        result = backend.translate("Test", "en", "en")
        assert result["text"] == "Hello"

    def test_translate_empty_content_list(self) -> None:
        """Translate handles empty content list."""
        client = FakeClientEmpty()
        backend = AnthropicBackend(client=client, model="test")
        result = backend.translate("Test", "en", "fr")
        assert result["text"] == ""

    def test_translate_skips_non_text_blocks(self) -> None:
        """Translate skips non-text content blocks."""
        client = FakeClientMixed()
        backend = AnthropicBackend(client=client, model="test")
        result = backend.translate("Test", "en", "fr")
        assert result["text"] == "Found text"

    def test_implements_protocol(self) -> None:
        """AnthropicBackend implements TranslationBackendProtocol."""
        client = FakeAnthropicClient()
        backend: TranslationBackendProtocol = AnthropicBackend(client=client, model="test")
        assert backend.backend_id == "anthropic"


class TestAnthropicTranslationSystemPrompt:
    """Tests for ANTHROPIC_SYSTEM_PROMPT constant."""

    def test_contains_rules(self) -> None:
        """System prompt contains translation rules."""
        assert "Translate" in ANTHROPIC_SYSTEM_PROMPT
        assert "ONLY" in ANTHROPIC_SYSTEM_PROMPT


class TestCreateAnthropicBackend:
    """Tests for create_anthropic_backend function."""

    def test_creates_backend(self) -> None:
        """Creates AnthropicBackend with given credentials."""
        from platform_translate.backends.anthropic import create_anthropic_backend

        backend = create_anthropic_backend(
            api_key="test-api-key",
            model="claude-3-haiku-20240307",
        )
        assert backend.backend_id == "anthropic"

    def test_created_backend_has_model(self) -> None:
        """Created backend uses provided model."""
        from platform_translate.backends.anthropic import create_anthropic_backend

        backend = create_anthropic_backend(
            api_key="test-key",
            model="claude-3-sonnet-20240229",
        )
        assert backend._model == "claude-3-sonnet-20240229"


# =============================================================================
# OpenAI Backend Tests
# =============================================================================


class FakeOpenAICompletionEmpty:
    """Fake completion with empty choices list."""

    __slots__ = ("choices",)

    def __init__(self) -> None:
        """Initialize with empty choices."""
        self.choices: list[ChoiceProtocol] = []


class FakeOpenAIMessageNone:
    """Fake message with None content."""

    __slots__ = ("content",)

    def __init__(self) -> None:
        """Initialize with None content."""
        self.content: str | None = None


class FakeOpenAIChoiceNone:
    """Fake choice with None content message."""

    __slots__ = ("message",)

    def __init__(self) -> None:
        """Initialize with None content message."""
        self.message = FakeOpenAIMessageNone()


class FakeOpenAICompletionNone:
    """Fake completion with None content in message."""

    __slots__ = ("choices",)

    def __init__(self) -> None:
        """Initialize with choice containing None content."""
        self.choices: list[ChoiceProtocol] = [FakeOpenAIChoiceNone()]


class FakeOpenAICompletionsEmpty:
    """Fake completions API that returns empty choices."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        """Initialize with empty calls list."""
        self.calls: list[dict[str, str | int | list[dict[str, str]]]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> CompletionProtocol:
        """Return completion with empty choices."""
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        return FakeOpenAICompletionEmpty()


class FakeOpenAICompletionsNone:
    """Fake completions API that returns None content."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        """Initialize with empty calls list."""
        self.calls: list[dict[str, str | int | list[dict[str, str]]]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> CompletionProtocol:
        """Return completion with None content."""
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        return FakeOpenAICompletionNone()


class FakeOpenAIChatEmpty:
    """Fake chat namespace that returns empty choices."""

    __slots__ = ("completions",)

    def __init__(self) -> None:
        """Initialize with empty completions."""
        self.completions = FakeOpenAICompletionsEmpty()


class FakeOpenAIChatNone:
    """Fake chat namespace that returns None content."""

    __slots__ = ("completions",)

    def __init__(self) -> None:
        """Initialize with None content completions."""
        self.completions = FakeOpenAICompletionsNone()


class FakeOpenAIClientEmpty:
    """Fake client that returns empty choices."""

    __slots__ = ("chat",)

    def __init__(self) -> None:
        """Initialize with empty chat."""
        self.chat = FakeOpenAIChatEmpty()


class FakeOpenAIClientNone:
    """Fake client that returns None content."""

    __slots__ = ("chat",)

    def __init__(self) -> None:
        """Initialize with None content chat."""
        self.chat = FakeOpenAIChatNone()


class TestOpenAIBackend:
    """Tests for OpenAIBackend class."""

    def test_backend_id(self) -> None:
        """Backend ID is 'openai'."""
        client = FakeOpenAIClient()
        backend = OpenAIBackend(client=client, model="test")
        assert backend.backend_id == "openai"
        assert OPENAI_BACKEND_ID == "openai"

    def test_translate_returns_result(self) -> None:
        """Translate returns TranslationResult."""
        client = FakeOpenAIClient(response_text="Hola")
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        result = backend.translate("Hello", "en", "es")
        assert result["text"] == "Hola"
        assert result["source_language"] == "en"
        assert result["target_language"] == "es"
        assert result["backend"] == "openai"

    def test_translate_calls_api_with_correct_params(self) -> None:
        """Translate calls API with correct parameters."""
        client = FakeOpenAIClient()
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        backend.translate("Test", "en", "fr")

        assert len(client.chat.completions.calls) == 1
        call = client.chat.completions.calls[0]
        assert call["model"] == "gpt-4o-mini"
        assert call["max_tokens"] == 4096

    def test_translate_includes_system_prompt(self) -> None:
        """Translate includes system prompt in messages."""
        client = FakeOpenAIClient()
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        backend.translate("Test", "en", "fr")

        messages = client.chat.completions.last_messages
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == OPENAI_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"

    def test_translate_empty_raises(self) -> None:
        """Translate raises ValueError for empty text."""
        client = FakeOpenAIClient()
        backend = OpenAIBackend(client=client, model="test")
        with pytest.raises(ValueError, match="cannot be empty"):
            backend.translate("", "en", "es")

    def test_translate_whitespace_raises(self) -> None:
        """Translate raises ValueError for whitespace-only text."""
        client = FakeOpenAIClient()
        backend = OpenAIBackend(client=client, model="test")
        with pytest.raises(ValueError, match="cannot be empty"):
            backend.translate("   ", "en", "es")

    def test_translate_strips_result(self) -> None:
        """Translate strips whitespace from result."""
        client = FakeOpenAIClient(response_text="  Hello  ")
        backend = OpenAIBackend(client=client, model="test")
        result = backend.translate("Test", "en", "en")
        assert result["text"] == "Hello"

    def test_translate_empty_choices_list(self) -> None:
        """Translate handles empty choices list."""
        client = FakeOpenAIClientEmpty()
        backend = OpenAIBackend(client=client, model="test")
        result = backend.translate("Test", "en", "fr")
        assert result["text"] == ""

    def test_translate_none_content(self) -> None:
        """Translate handles None content in message."""
        client = FakeOpenAIClientNone()
        backend = OpenAIBackend(client=client, model="test")
        result = backend.translate("Test", "en", "fr")
        assert result["text"] == ""

    def test_implements_protocol(self) -> None:
        """OpenAIBackend implements TranslationBackendProtocol."""
        client = FakeOpenAIClient()
        backend: TranslationBackendProtocol = OpenAIBackend(client=client, model="test")
        assert backend.backend_id == "openai"


class TestOpenAITranslationSystemPrompt:
    """Tests for OPENAI_SYSTEM_PROMPT constant."""

    def test_contains_rules(self) -> None:
        """System prompt contains translation rules."""
        assert "Translate" in OPENAI_SYSTEM_PROMPT
        assert "ONLY" in OPENAI_SYSTEM_PROMPT


class TestCreateOpenAIBackend:
    """Tests for create_openai_backend function."""

    def test_creates_backend(self) -> None:
        """Creates OpenAIBackend with given credentials."""
        from platform_translate.backends.openai import create_openai_backend

        backend = create_openai_backend(
            api_key="test-api-key",
            model="gpt-4o-mini",
        )
        assert backend.backend_id == "openai"

    def test_created_backend_has_model(self) -> None:
        """Created backend uses provided model."""
        from platform_translate.backends.openai import create_openai_backend

        backend = create_openai_backend(
            api_key="test-key",
            model="gpt-4o",
        )
        assert backend._model == "gpt-4o"
