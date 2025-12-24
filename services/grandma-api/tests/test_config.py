"""Tests for grandma_api.config module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from grandma_api.config import (
    GrandmaApiSettings,
    _validate_log_format,
    _validate_log_level,
    decode_grandma_api_settings,
    encode_grandma_api_settings,
    load_settings,
    require_grandma_api_settings,
)

from .conftest import set_fake_env


def test_encode_grandma_api_settings() -> None:
    """Test encoding GrandmaApiSettings to JSON."""
    settings = GrandmaApiSettings(
        openai_api_key="sk-test",
        api_token="secret",
        port=8080,
        log_level="INFO",
        log_format="json",
    )
    encoded = encode_grandma_api_settings(settings)
    assert encoded["openai_api_key"] == "sk-test"
    assert encoded["api_token"] == "secret"
    assert encoded["port"] == 8080
    assert encoded["log_level"] == "INFO"
    assert encoded["log_format"] == "json"


def test_decode_grandma_api_settings() -> None:
    """Test decoding JSON to GrandmaApiSettings."""
    obj: JSONObject = {
        "openai_api_key": "sk-test",
        "api_token": "secret",
        "port": 8080,
        "log_level": "DEBUG",
        "log_format": "text",
    }
    decoded = decode_grandma_api_settings(obj)
    assert decoded["openai_api_key"] == "sk-test"
    assert decoded["api_token"] == "secret"
    assert decoded["port"] == 8080
    assert decoded["log_level"] == "DEBUG"
    assert decoded["log_format"] == "text"


def test_decode_grandma_api_settings_missing_field() -> None:
    """Test decode raises when required field is missing."""
    obj: JSONObject = {"openai_api_key": "sk-test"}
    with pytest.raises(JSONTypeError, match="Missing required field"):
        decode_grandma_api_settings(obj)


def test_require_grandma_api_settings() -> None:
    """Test require_grandma_api_settings with valid dict."""
    obj: JSONValue = {
        "openai_api_key": "sk-test",
        "api_token": "secret",
        "port": 8080,
        "log_level": "INFO",
        "log_format": "json",
    }
    result = require_grandma_api_settings(obj)
    assert result["openai_api_key"] == "sk-test"


def test_require_grandma_api_settings_not_dict() -> None:
    """Test require_grandma_api_settings raises when not a dict."""
    value: JSONValue = "not a dict"
    with pytest.raises(JSONTypeError, match="Expected object"):
        require_grandma_api_settings(value)


def test_validate_log_level_all_levels() -> None:
    """Test _validate_log_level with all valid levels."""
    assert _validate_log_level("debug") == "DEBUG"
    assert _validate_log_level("INFO") == "INFO"
    assert _validate_log_level("Warning") == "WARNING"
    assert _validate_log_level("ERROR") == "ERROR"
    assert _validate_log_level("critical") == "CRITICAL"


def test_validate_log_level_invalid() -> None:
    """Test _validate_log_level raises on invalid level."""
    with pytest.raises(JSONTypeError, match="Invalid log level"):
        _validate_log_level("INVALID")


def test_validate_log_format_all_formats() -> None:
    """Test _validate_log_format with all valid formats."""
    assert _validate_log_format("JSON") == "json"
    assert _validate_log_format("text") == "text"


def test_validate_log_format_invalid() -> None:
    """Test _validate_log_format raises on invalid format."""
    with pytest.raises(JSONTypeError, match="Invalid log format"):
        _validate_log_format("xml")


def test_load_settings_success() -> None:
    """Test loading settings from environment."""
    set_fake_env(
        {
            "OPENAI_API_KEY": "sk-test-key",
            "API_TOKEN": "my-secret-token",
            "PORT": "9000",
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "text",
        }
    )

    settings = load_settings()

    assert settings["openai_api_key"] == "sk-test-key"
    assert settings["api_token"] == "my-secret-token"
    assert settings["port"] == 9000
    assert settings["log_level"] == "DEBUG"
    assert settings["log_format"] == "text"


def test_load_settings_defaults() -> None:
    """Test loading settings uses defaults for optional vars."""
    set_fake_env(
        {
            "OPENAI_API_KEY": "sk-test-key",
            "API_TOKEN": "my-secret-token",
        }
    )

    settings = load_settings()

    assert settings["port"] == 8080
    assert settings["log_level"] == "INFO"
    assert settings["log_format"] == "json"


def test_load_settings_missing_openai_key() -> None:
    """Test load_settings raises when OPENAI_API_KEY is missing."""
    set_fake_env({"API_TOKEN": "token"})

    with pytest.raises(RuntimeError, match="Missing required env var: OPENAI_API_KEY"):
        load_settings()


def test_load_settings_missing_api_token() -> None:
    """Test load_settings raises when API_TOKEN is missing."""
    set_fake_env({"OPENAI_API_KEY": "sk-test"})

    with pytest.raises(RuntimeError, match="Missing required env var: API_TOKEN"):
        load_settings()


def test_load_settings_empty_openai_key() -> None:
    """Test load_settings raises when OPENAI_API_KEY is empty."""
    set_fake_env({"OPENAI_API_KEY": "  ", "API_TOKEN": "token"})

    with pytest.raises(RuntimeError, match="Empty env var: OPENAI_API_KEY"):
        load_settings()


def test_load_settings_invalid_log_level_uses_default() -> None:
    """Test invalid log level falls back to default."""
    set_fake_env(
        {
            "OPENAI_API_KEY": "sk-test",
            "API_TOKEN": "token",
            "LOG_LEVEL": "INVALID",
        }
    )

    settings = load_settings()
    assert settings["log_level"] == "INFO"


def test_load_settings_invalid_log_format_uses_default() -> None:
    """Test invalid log format falls back to default."""
    set_fake_env(
        {
            "OPENAI_API_KEY": "sk-test",
            "API_TOKEN": "token",
            "LOG_FORMAT": "xml",
        }
    )

    settings = load_settings()
    assert settings["log_format"] == "json"


def test_load_settings_explicit_json_format() -> None:
    """Test loading settings with explicit json format."""
    set_fake_env(
        {
            "OPENAI_API_KEY": "sk-test",
            "API_TOKEN": "token",
            "LOG_FORMAT": "json",
        }
    )

    settings = load_settings()
    assert settings["log_format"] == "json"
