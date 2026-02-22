"""Tests for platform_translate.translator module."""

from __future__ import annotations

import pytest

from platform_translate import _test_hooks
from platform_translate.testing import (
    make_fake_backend_factory,
    reset_hooks,
)
from platform_translate.translator import Translator, create_translator, translate_text
from platform_translate.types import TranslatorConfig


class TestTranslator:
    """Tests for Translator class."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()
        _test_hooks.backend_factory = make_fake_backend_factory(
            translated_text="Hello",
            backend_id="test-backend",
        )

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_translate_returns_result(self) -> None:
        """Translate returns TranslationResult."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="test",
            model="test",
        )
        translator = Translator(config)
        result = translator.translate("Xin chào", "vi", "en")
        assert result["text"] == "Hello"
        assert result["source_language"] == "vi"
        assert result["target_language"] == "en"
        assert result["backend"] == "test-backend"

    def test_translate_empty_raises(self) -> None:
        """Translate raises ValueError for empty text."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="test",
            model="test",
        )
        translator = Translator(config)
        with pytest.raises(ValueError, match="cannot be empty"):
            translator.translate("", "vi", "en")

    def test_translate_whitespace_raises(self) -> None:
        """Translate raises ValueError for whitespace-only text."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="test",
            model="test",
        )
        translator = Translator(config)
        with pytest.raises(ValueError, match="cannot be empty"):
            translator.translate("   ", "vi", "en")

    def test_backend_id_property(self) -> None:
        """Backend ID property returns backend identifier."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="test",
            model="test",
        )
        translator = Translator(config)
        assert translator.backend_id == "test-backend"


class TestTranslateText:
    """Tests for translate_text convenience function."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()
        _test_hooks.backend_factory = make_fake_backend_factory(
            translated_text="Bonjour",
            backend_id="fake",
        )

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_translate_text_returns_result(self) -> None:
        """translate_text returns TranslationResult."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="test",
            model="test",
        )
        result = translate_text("Hello", "en", "fr", config)
        assert result["text"] == "Bonjour"
        assert result["source_language"] == "en"
        assert result["target_language"] == "fr"


class TestCreateTranslator:
    """Tests for create_translator function."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()
        _test_hooks.backend_factory = make_fake_backend_factory()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_create_returns_translator(self) -> None:
        """create_translator returns working translator."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="test",
            model="test",
        )
        translator = create_translator(config)
        # Verify translator has backend_id property (Translator-specific)
        assert translator.backend_id == "fake"

    def test_create_translator_works(self) -> None:
        """Created translator can translate."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="test",
            model="test",
        )
        translator = create_translator(config)
        result = translator.translate("Test", "en", "es")
        assert "text" in result
