"""Tests for configuration loading."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from covenant_radar_api.core import Settings, settings_from_env
from covenant_radar_api.core.config import Settings as ConfigSettings


def test_settings_from_env_loads_required_vars(monkeypatch: MonkeyPatch) -> None:
    """Test settings_from_env loads from environment variables."""
    monkeypatch.setenv("REDIS_URL", "redis://test-redis:6379/0")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")

    s = settings_from_env()

    assert s["redis_url"] == "redis://test-redis:6379/0"
    assert s["database_url"] == "postgresql://user:pass@host/db"


def test_settings_from_env_missing_redis_url(monkeypatch: MonkeyPatch) -> None:
    """Test settings_from_env raises when REDIS_URL missing."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")

    with pytest.raises(RuntimeError, match="REDIS_URL"):
        settings_from_env()


def test_settings_from_env_missing_database_url(monkeypatch: MonkeyPatch) -> None:
    """Test settings_from_env raises when DATABASE_URL missing."""
    monkeypatch.setenv("REDIS_URL", "redis://test")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        settings_from_env()


def test_settings_type_is_typed_dict() -> None:
    """Test Settings type is a TypedDict."""
    assert Settings is ConfigSettings
