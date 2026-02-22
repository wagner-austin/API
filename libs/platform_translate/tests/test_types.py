"""Tests for platform_translate.types module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_translate.types import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    TranslationRequest,
    TranslationResult,
    TranslatorConfig,
    decode_translation_request,
    decode_translation_result,
    decode_translator_config,
    default_translator_config,
    encode_translation_request,
    encode_translation_result,
    encode_translator_config,
    require_translation_request,
    require_translation_result,
    require_translator_config,
)


class TestTranslationRequest:
    """Tests for TranslationRequest TypedDict."""

    def test_encode_basic(self) -> None:
        """Encode basic request."""
        request = TranslationRequest(
            text="Xin chào",
            source_language="vi",
            target_language="en",
        )
        encoded = encode_translation_request(request)
        assert encoded == {
            "text": "Xin chào",
            "source_language": "vi",
            "target_language": "en",
        }

    def test_decode_valid(self) -> None:
        """Decode valid request."""
        obj: JSONObject = {
            "text": "Hello",
            "source_language": "en",
            "target_language": "es",
        }
        request = decode_translation_request(obj)
        assert request["text"] == "Hello"
        assert request["source_language"] == "en"
        assert request["target_language"] == "es"

    def test_decode_missing_text(self) -> None:
        """Raise for missing text field."""
        obj: JSONObject = {"source_language": "en", "target_language": "es"}
        with pytest.raises(JSONTypeError):
            decode_translation_request(obj)

    def test_decode_missing_source_language(self) -> None:
        """Raise for missing source_language field."""
        obj: JSONObject = {"text": "Hello", "target_language": "es"}
        with pytest.raises(JSONTypeError):
            decode_translation_request(obj)

    def test_decode_missing_target_language(self) -> None:
        """Raise for missing target_language field."""
        obj: JSONObject = {"text": "Hello", "source_language": "en"}
        with pytest.raises(JSONTypeError):
            decode_translation_request(obj)

    def test_decode_empty_text(self) -> None:
        """Raise for empty text."""
        obj: JSONObject = {
            "text": "",
            "source_language": "en",
            "target_language": "es",
        }
        with pytest.raises(ValueError, match="cannot be empty"):
            decode_translation_request(obj)

    def test_decode_whitespace_only_text(self) -> None:
        """Raise for whitespace-only text."""
        obj: JSONObject = {
            "text": "   ",
            "source_language": "en",
            "target_language": "es",
        }
        with pytest.raises(ValueError, match="cannot be empty"):
            decode_translation_request(obj)

    def test_require_valid(self) -> None:
        """Require valid request from JSONValue."""
        obj: JSONObject = {
            "text": "Test",
            "source_language": "en",
            "target_language": "fr",
        }
        request = require_translation_request(obj)
        assert request["text"] == "Test"

    def test_require_non_dict(self) -> None:
        """Raise for non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_translation_request("not a dict")

    def test_roundtrip(self) -> None:
        """Encode then decode produces identical request."""
        original = TranslationRequest(
            text="Bonjour",
            source_language="fr",
            target_language="de",
        )
        encoded = encode_translation_request(original)
        decoded = decode_translation_request(encoded)
        assert decoded == original


class TestTranslationResult:
    """Tests for TranslationResult TypedDict."""

    def test_encode_basic(self) -> None:
        """Encode basic result."""
        result = TranslationResult(
            text="Hello",
            source_language="vi",
            target_language="en",
            backend="anthropic",
        )
        encoded = encode_translation_result(result)
        assert encoded == {
            "text": "Hello",
            "source_language": "vi",
            "target_language": "en",
            "backend": "anthropic",
        }

    def test_decode_valid(self) -> None:
        """Decode valid result."""
        obj: JSONObject = {
            "text": "Hola",
            "source_language": "en",
            "target_language": "es",
            "backend": "deepl",
        }
        result = decode_translation_result(obj)
        assert result["text"] == "Hola"
        assert result["backend"] == "deepl"

    def test_decode_missing_text(self) -> None:
        """Raise for missing text field."""
        obj: JSONObject = {
            "source_language": "en",
            "target_language": "es",
            "backend": "test",
        }
        with pytest.raises(JSONTypeError):
            decode_translation_result(obj)

    def test_decode_missing_backend(self) -> None:
        """Raise for missing backend field."""
        obj: JSONObject = {
            "text": "Test",
            "source_language": "en",
            "target_language": "es",
        }
        with pytest.raises(JSONTypeError):
            decode_translation_result(obj)

    def test_require_valid(self) -> None:
        """Require valid result from JSONValue."""
        obj: JSONObject = {
            "text": "Test",
            "source_language": "en",
            "target_language": "fr",
            "backend": "nllb",
        }
        result = require_translation_result(obj)
        assert result["backend"] == "nllb"

    def test_require_non_dict(self) -> None:
        """Raise for non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_translation_result([1, 2, 3])

    def test_roundtrip(self) -> None:
        """Encode then decode produces identical result."""
        original = TranslationResult(
            text="Bonjour",
            source_language="en",
            target_language="fr",
            backend="anthropic",
        )
        encoded = encode_translation_result(original)
        decoded = decode_translation_result(encoded)
        assert decoded == original


class TestTranslatorConfig:
    """Tests for TranslatorConfig TypedDict."""

    def test_encode_basic(self) -> None:
        """Encode basic config."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="sk-test",
            model="claude-3-haiku",
        )
        encoded = encode_translator_config(config)
        assert encoded == {
            "backend": "anthropic",
            "api_key": "sk-test",
            "model": "claude-3-haiku",
        }

    def test_decode_valid_anthropic(self) -> None:
        """Decode valid anthropic config."""
        obj: JSONObject = {
            "backend": "anthropic",
            "api_key": "key",
            "model": "model",
        }
        config = decode_translator_config(obj)
        assert config["backend"] == "anthropic"

    def test_decode_valid_deepl(self) -> None:
        """Decode valid deepl config."""
        obj: JSONObject = {
            "backend": "deepl",
            "api_key": "key",
            "model": "model",
        }
        config = decode_translator_config(obj)
        assert config["backend"] == "deepl"

    def test_decode_valid_nllb(self) -> None:
        """Decode valid nllb config."""
        obj: JSONObject = {
            "backend": "nllb",
            "api_key": "key",
            "model": "model",
        }
        config = decode_translator_config(obj)
        assert config["backend"] == "nllb"

    def test_decode_missing_backend(self) -> None:
        """Raise for missing backend field."""
        obj: JSONObject = {"api_key": "key", "model": "model"}
        with pytest.raises(JSONTypeError):
            decode_translator_config(obj)

    def test_decode_missing_api_key(self) -> None:
        """Raise for missing api_key field."""
        obj: JSONObject = {"backend": "anthropic", "model": "model"}
        with pytest.raises(JSONTypeError):
            decode_translator_config(obj)

    def test_decode_missing_model(self) -> None:
        """Raise for missing model field."""
        obj: JSONObject = {"backend": "anthropic", "api_key": "key"}
        with pytest.raises(JSONTypeError):
            decode_translator_config(obj)

    def test_decode_unsupported_backend(self) -> None:
        """Raise for unsupported backend."""
        obj: JSONObject = {
            "backend": "unsupported",
            "api_key": "key",
            "model": "model",
        }
        with pytest.raises(ValueError, match="must be one of"):
            decode_translator_config(obj)

    def test_require_valid(self) -> None:
        """Require valid config from JSONValue."""
        obj: JSONObject = {
            "backend": "anthropic",
            "api_key": "key",
            "model": "model",
        }
        config = require_translator_config(obj)
        assert config["backend"] == "anthropic"

    def test_require_non_dict(self) -> None:
        """Raise for non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_translator_config(123)

    def test_roundtrip(self) -> None:
        """Encode then decode produces identical config."""
        original = TranslatorConfig(
            backend="deepl",
            api_key="secret-key",
            model="deepl-pro",
        )
        encoded = encode_translator_config(original)
        decoded = decode_translator_config(encoded)
        assert decoded == original


class TestDefaultConfig:
    """Tests for default_translator_config function."""

    def test_returns_correct_defaults(self) -> None:
        """Returns config with default values."""
        config = default_translator_config("test-key")
        assert config["backend"] == DEFAULT_BACKEND
        assert config["api_key"] == "test-key"
        assert config["model"] == DEFAULT_MODEL

    def test_default_backend_value(self) -> None:
        """DEFAULT_BACKEND has expected value."""
        assert DEFAULT_BACKEND == "anthropic"

    def test_default_model_value(self) -> None:
        """DEFAULT_MODEL has expected value."""
        assert DEFAULT_MODEL == "claude-3-haiku-20240307"
