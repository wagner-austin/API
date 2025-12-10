"""Tests for covenant_radar configuration loading."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from platform_core.config import CovenantRadarSettings, load_covenant_radar_settings


def test_load_covenant_radar_settings_success(monkeypatch: MonkeyPatch) -> None:
    """Test loading covenant radar settings from environment."""
    monkeypatch.setenv("REDIS_URL", "redis://test-redis:6379/0")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")

    settings = load_covenant_radar_settings()

    assert settings["redis_url"] == "redis://test-redis:6379/0"
    assert settings["database_url"] == "postgresql://user:pass@host:5432/db"


def test_load_covenant_radar_settings_missing_redis_url(monkeypatch: MonkeyPatch) -> None:
    """Test load_covenant_radar_settings raises when REDIS_URL missing."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")

    with pytest.raises(RuntimeError, match="REDIS_URL"):
        load_covenant_radar_settings()


def test_load_covenant_radar_settings_missing_database_url(monkeypatch: MonkeyPatch) -> None:
    """Test load_covenant_radar_settings raises when DATABASE_URL missing."""
    monkeypatch.setenv("REDIS_URL", "redis://test")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        load_covenant_radar_settings()


def test_covenant_radar_settings_is_typed_dict() -> None:
    """Test CovenantRadarSettings is a proper TypedDict."""
    annotations = CovenantRadarSettings.__annotations__
    assert "redis_url" in annotations
    assert "database_url" in annotations
