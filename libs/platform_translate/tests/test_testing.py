"""Tests for platform_translate.testing module."""

from __future__ import annotations

import pytest

from platform_translate import _test_hooks
from platform_translate.testing import (
    FakeAnthropicClient,
    FakeMessage,
    FakeMessageContent,
    FakeMessages,
    FakeTranslationBackend,
    make_fake_backend_factory,
    reset_hooks,
    set_production_hooks,
)
from platform_translate.types import TranslatorConfig


class TestFakeMessageContent:
    """Tests for FakeMessageContent class."""

    def test_has_text(self) -> None:
        """Has text attribute."""
        content = FakeMessageContent("Hello")
        assert content.text == "Hello"

    def test_has_type(self) -> None:
        """Has type attribute."""
        content = FakeMessageContent("Hello", "text")
        assert content.type == "text"

    def test_default_type(self) -> None:
        """Default type is 'text'."""
        content = FakeMessageContent("Test")
        assert content.type == "text"


class TestFakeMessage:
    """Tests for FakeMessage class."""

    def test_has_content(self) -> None:
        """Has content list with text block."""
        message = FakeMessage("Hello")
        assert len(message.content) == 1
        assert message.content[0].text == "Hello"
        assert message.content[0].type == "text"


class TestFakeMessages:
    """Tests for FakeMessages class."""

    def test_create_returns_message(self) -> None:
        """Create returns FakeMessage with expected content."""
        messages = FakeMessages(response_text="Bonjour")
        result = messages.create(
            model="test",
            max_tokens=100,
            system="translate",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # Verify result has expected structure and content
        assert result.content[0].text == "Bonjour"
        assert result.content[0].type == "text"

    def test_create_records_calls(self) -> None:
        """Create records all calls."""
        messages = FakeMessages()
        messages.create(
            model="model1",
            max_tokens=100,
            system="sys1",
            messages=[{"role": "user", "content": "test1"}],
        )
        messages.create(
            model="model2",
            max_tokens=200,
            system="sys2",
            messages=[{"role": "user", "content": "test2"}],
        )

        assert len(messages.calls) == 2
        assert messages.calls[0]["model"] == "model1"
        assert messages.calls[1]["model"] == "model2"


class TestFakeAnthropicClient:
    """Tests for FakeAnthropicClient class."""

    def test_has_messages(self) -> None:
        """Messages attribute returns FakeMessages instance."""
        client = FakeAnthropicClient()
        # Verify messages can be called
        result = client.messages.create(
            model="test",
            max_tokens=100,
            system="sys",
            messages=[{"role": "user", "content": "test"}],
        )
        assert result.content[0].type == "text"

    def test_messages_create_works(self) -> None:
        """Messages create returns response."""
        client = FakeAnthropicClient(response_text="Test")
        result = client.messages.create(
            model="test",
            max_tokens=100,
            system="sys",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result.content[0].text == "Test"


class TestFakeTranslationBackend:
    """Tests for FakeTranslationBackend class."""

    def test_translate_returns_configured_text(self) -> None:
        """Translate returns configured text."""
        backend = FakeTranslationBackend(translated_text="Bonjour")
        result = backend.translate("Hello", "en", "fr")
        assert result["text"] == "Bonjour"

    def test_translate_returns_languages(self) -> None:
        """Translate returns source and target languages."""
        backend = FakeTranslationBackend()
        result = backend.translate("Test", "vi", "en")
        assert result["source_language"] == "vi"
        assert result["target_language"] == "en"

    def test_translate_returns_backend_id(self) -> None:
        """Translate returns configured backend_id."""
        backend = FakeTranslationBackend(backend_id="test-backend")
        result = backend.translate("Test", "en", "es")
        assert result["backend"] == "test-backend"

    def test_backend_id_property(self) -> None:
        """Backend ID property returns configured value."""
        backend = FakeTranslationBackend(backend_id="custom")
        assert backend.backend_id == "custom"

    def test_translate_empty_raises(self) -> None:
        """Translate raises for empty text."""
        backend = FakeTranslationBackend()
        with pytest.raises(ValueError, match="cannot be empty"):
            backend.translate("", "en", "es")

    def test_translate_increments_call_count(self) -> None:
        """Translate increments call_count."""
        backend = FakeTranslationBackend()
        assert backend.call_count == 0
        backend.translate("Test", "en", "es")
        assert backend.call_count == 1
        backend.translate("Test", "en", "fr")
        assert backend.call_count == 2


class TestHookManagement:
    """Tests for hook management functions."""

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_set_production_hooks(self) -> None:
        """set_production_hooks restores defaults."""
        _test_hooks.backend_factory = make_fake_backend_factory()

        set_production_hooks()

        assert _test_hooks.backend_factory is _test_hooks._default_backend_factory

    def test_reset_hooks(self) -> None:
        """reset_hooks restores defaults."""
        _test_hooks.backend_factory = make_fake_backend_factory()

        reset_hooks()

        assert _test_hooks.backend_factory is _test_hooks._default_backend_factory


class TestMakeFakeBackendFactory:
    """Tests for make_fake_backend_factory function."""

    def test_creates_factory(self) -> None:
        """Creates factory that returns FakeTranslationBackend."""
        factory = make_fake_backend_factory(
            translated_text="Hola",
            backend_id="fake",
        )
        config = TranslatorConfig(
            backend="anthropic",
            api_key="key",
            model="model",
        )
        backend = factory(config)
        result = backend.translate("Hello", "en", "es")
        assert result["text"] == "Hola"
        assert result["backend"] == "fake"
