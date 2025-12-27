"""Tests for config module."""

from __future__ import annotations

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, JSONTypeError

from opportunity_radar_api.config import (
    OpportunityRadarSettings,
    decode_opportunity_radar_settings,
    encode_opportunity_radar_settings,
    load_settings,
    require_opportunity_radar_settings,
)


class TestEncodeDecodeSettings:
    """Tests for encode/decode round-trip."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test that encoding then decoding returns same settings."""
        settings = OpportunityRadarSettings(
            kaggle_api_token="test-token",
            port=8080,
            log_level="DEBUG",
            log_format="text",
            github_token="gh-token",
            github_repo="owner/repo",
        )

        encoded = encode_opportunity_radar_settings(settings)
        decoded = decode_opportunity_radar_settings(encoded)

        assert decoded["kaggle_api_token"] == "test-token"
        assert decoded["port"] == 8080
        assert decoded["log_level"] == "DEBUG"
        assert decoded["log_format"] == "text"
        assert decoded["github_token"] == "gh-token"
        assert decoded["github_repo"] == "owner/repo"

    def test_encode_decode_with_none_github(self) -> None:
        """Test encoding/decoding with None github settings."""
        settings = OpportunityRadarSettings(
            kaggle_api_token="",
            port=8010,
            log_level="INFO",
            log_format="json",
            github_token=None,
            github_repo=None,
        )

        encoded = encode_opportunity_radar_settings(settings)
        decoded = decode_opportunity_radar_settings(encoded)

        assert decoded["github_token"] is None
        assert decoded["github_repo"] is None


class TestDecodeSettings:
    """Tests for decode_opportunity_radar_settings."""

    def test_decode_all_log_levels(self) -> None:
        """Test decoding all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            data: JSONObject = {
                "kaggle_api_token": "",
                "port": 8010,
                "log_level": level,
                "log_format": "json",
                "github_token": None,
                "github_repo": None,
            }
            settings = decode_opportunity_radar_settings(data)
            assert settings["log_level"] == level

    def test_decode_log_levels_case_insensitive(self) -> None:
        """Test that log level decoding is case insensitive."""
        data: JSONObject = {
            "kaggle_api_token": "",
            "port": 8010,
            "log_level": "debug",
            "log_format": "json",
            "github_token": None,
            "github_repo": None,
        }
        settings = decode_opportunity_radar_settings(data)
        assert settings["log_level"] == "DEBUG"

    def test_decode_invalid_log_level(self) -> None:
        """Test that invalid log level raises error."""
        data: JSONObject = {
            "kaggle_api_token": "",
            "port": 8010,
            "log_level": "INVALID",
            "log_format": "json",
            "github_token": None,
            "github_repo": None,
        }
        with pytest.raises(JSONTypeError, match="Invalid log level"):
            decode_opportunity_radar_settings(data)

    def test_decode_all_log_formats(self) -> None:
        """Test decoding all valid log formats."""
        for fmt in ["json", "text"]:
            data: JSONObject = {
                "kaggle_api_token": "",
                "port": 8010,
                "log_level": "INFO",
                "log_format": fmt,
                "github_token": None,
                "github_repo": None,
            }
            settings = decode_opportunity_radar_settings(data)
            assert settings["log_format"] == fmt

    def test_decode_log_format_case_insensitive(self) -> None:
        """Test that log format decoding is case insensitive."""
        data: JSONObject = {
            "kaggle_api_token": "",
            "port": 8010,
            "log_level": "INFO",
            "log_format": "JSON",
            "github_token": None,
            "github_repo": None,
        }
        settings = decode_opportunity_radar_settings(data)
        assert settings["log_format"] == "json"

    def test_decode_invalid_log_format(self) -> None:
        """Test that invalid log format raises error."""
        data: JSONObject = {
            "kaggle_api_token": "",
            "port": 8010,
            "log_level": "INFO",
            "log_format": "xml",
            "github_token": None,
            "github_repo": None,
        }
        with pytest.raises(JSONTypeError, match="Invalid log format"):
            decode_opportunity_radar_settings(data)


class TestRequireSettings:
    """Tests for require_opportunity_radar_settings."""

    def test_require_with_valid_object(self) -> None:
        """Test requiring valid settings object."""
        data: JSONObject = {
            "kaggle_api_token": "token",
            "port": 8010,
            "log_level": "INFO",
            "log_format": "json",
            "github_token": None,
            "github_repo": None,
        }
        settings = require_opportunity_radar_settings(data)
        assert settings["kaggle_api_token"] == "token"

    def test_require_with_non_object(self) -> None:
        """Test that non-object raises error."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_opportunity_radar_settings("not an object")

    def test_require_with_list(self) -> None:
        """Test that list raises error."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_opportunity_radar_settings([])


class TestLoadSettings:
    """Tests for load_settings from environment."""

    def test_load_settings_defaults(self) -> None:
        """Test loading settings with defaults."""
        original_fn = config_test_hooks.get_env

        def fake_get_env(key: str) -> str | None:
            return None

        config_test_hooks.get_env = fake_get_env

        try:
            settings = load_settings()
            assert settings["kaggle_api_token"] == ""
            assert settings["port"] == 8010
            assert settings["log_level"] == "INFO"
            assert settings["log_format"] == "json"
            assert settings["github_token"] is None
            assert settings["github_repo"] is None
        finally:
            config_test_hooks.get_env = original_fn

    def test_load_settings_from_env(self) -> None:
        """Test loading settings from environment variables."""
        original_fn = config_test_hooks.get_env

        env_vars = {
            "KAGGLE_API_TOKEN": "my-kaggle-token",
            "PORT": "9000",
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "text",
            "GITHUB_TOKEN": "ghp_test",
            "GITHUB_REPO": "user/repo",
        }

        def fake_get_env(key: str) -> str | None:
            return env_vars.get(key)

        config_test_hooks.get_env = fake_get_env

        try:
            settings = load_settings()
            assert settings["kaggle_api_token"] == "my-kaggle-token"
            assert settings["port"] == 9000
            assert settings["log_level"] == "DEBUG"
            assert settings["log_format"] == "text"
            assert settings["github_token"] == "ghp_test"
            assert settings["github_repo"] == "user/repo"
        finally:
            config_test_hooks.get_env = original_fn

    def test_load_settings_log_format_defaults_on_invalid(self) -> None:
        """Test that invalid log format falls back to default."""
        original_fn = config_test_hooks.get_env

        env_vars = {
            "LOG_FORMAT": "invalid_format",
        }

        def fake_get_env(key: str) -> str | None:
            return env_vars.get(key)

        config_test_hooks.get_env = fake_get_env

        try:
            settings = load_settings()
            assert settings["log_format"] == "json"  # Default
        finally:
            config_test_hooks.get_env = original_fn

    def test_load_settings_log_format_json_explicit(self) -> None:
        """Test loading settings with explicit json log format."""
        original_fn = config_test_hooks.get_env

        env_vars = {
            "LOG_FORMAT": "json",
        }

        def fake_get_env(key: str) -> str | None:
            return env_vars.get(key)

        config_test_hooks.get_env = fake_get_env

        try:
            settings = load_settings()
            assert settings["log_format"] == "json"
        finally:
            config_test_hooks.get_env = original_fn
