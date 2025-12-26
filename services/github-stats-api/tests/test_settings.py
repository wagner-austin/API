from __future__ import annotations

from platform_core.config import config_test_hooks

from github_stats_api.settings import load_settings


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_settings_defaults(self) -> None:
        """Test loading settings with defaults."""
        original_hook = config_test_hooks.get_env

        def fake_get_env(key: str) -> str | None:
            # Return None for all keys to test defaults
            return None

        config_test_hooks.get_env = fake_get_env
        settings = load_settings()
        config_test_hooks.get_env = original_hook

        assert settings["github_token"] == ""
        assert settings["cache_ttl_seconds"] == 1800
        assert settings["port"] == 8000

    def test_load_settings_from_env(self) -> None:
        """Test loading settings from environment variables."""
        original_hook = config_test_hooks.get_env

        env_values: dict[str, str] = {
            "GITHUB_TOKEN": "ghp_testtoken",
            "CACHE_TTL_SECONDS": "3600",
            "PORT": "9000",
        }

        def fake_get_env(key: str) -> str | None:
            return env_values.get(key)

        config_test_hooks.get_env = fake_get_env
        settings = load_settings()
        config_test_hooks.get_env = original_hook

        assert settings["github_token"] == "ghp_testtoken"
        assert settings["cache_ttl_seconds"] == 3600
        assert settings["port"] == 9000

    def test_load_settings_partial_env(self) -> None:
        """Test loading settings with partial environment variables."""
        original_hook = config_test_hooks.get_env

        env_values: dict[str, str] = {
            "GITHUB_TOKEN": "test-token",
        }

        def fake_get_env(key: str) -> str | None:
            return env_values.get(key)

        config_test_hooks.get_env = fake_get_env
        settings = load_settings()
        config_test_hooks.get_env = original_hook

        assert settings["github_token"] == "test-token"
        assert settings["cache_ttl_seconds"] == 1800
        assert settings["port"] == 8000
