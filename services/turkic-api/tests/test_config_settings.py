"""Tests for turkic-api config settings loading."""

from __future__ import annotations

import pytest
from platform_core.config import config_test_hooks
from platform_core.testing import make_fake_env

from turkic_api.api.config import settings_from_env


def test_missing_api_key_raises() -> None:
    """Test that missing TURKIC_DATA_BANK_API_KEY raises RuntimeError."""
    env = make_fake_env(
        {
            "TURKIC_REDIS_URL": "redis://test:6379/0",
            "TURKIC_DATA_DIR": "/data",
            "TURKIC_DATA_BANK_API_URL": "http://db",
            # Missing TURKIC_DATA_BANK_API_KEY
        }
    )
    config_test_hooks.get_env = env

    with pytest.raises(RuntimeError):
        settings_from_env()


def test_defaults_and_overrides() -> None:
    """Test defaults are used when env vars missing, and overrides work."""
    # Minimal env - only required key
    env = make_fake_env(
        {
            "TURKIC_DATA_BANK_API_KEY": "secret",
        }
    )
    config_test_hooks.get_env = env

    cfg = settings_from_env()
    assert cfg["redis_url"] == "redis://redis:6379/0"
    assert cfg["data_dir"] == "/data"
    assert cfg["environment"] == "local"
    assert cfg["data_bank_api_url"] == ""
    assert cfg["data_bank_api_key"] == "secret"

    # Now with override
    env_with_override = make_fake_env(
        {
            "TURKIC_DATA_BANK_API_KEY": "secret",
            "TURKIC_REDIS_URL": "redis://override",
        }
    )
    config_test_hooks.get_env = env_with_override

    cfg2 = settings_from_env()
    assert cfg2["redis_url"] == "redis://override"
