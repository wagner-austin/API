"""Pytest configuration and fixtures for DiscordBot tests."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Generator

import pytest
from tests.support.settings import (
    SettingsFactory,
    build_settings,
    make_settings_factory,
)

from clubbot.config import DiscordbotSettings


def _settings_factory() -> SettingsFactory:
    return make_settings_factory()


settings_factory = pytest.fixture(name="settings_factory")(_settings_factory)


def _settings() -> DiscordbotSettings:
    return build_settings()


settings = pytest.fixture(name="settings")(_settings)


def _patch_load_settings_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Autouse fixture that patches load_discordbot_settings in all modules."""
    test_settings = build_settings()

    def _mock_load_settings() -> DiscordbotSettings:
        return test_settings

    monkeypatch.setattr("clubbot.config.load_discordbot_settings", _mock_load_settings)
    monkeypatch.setattr("clubbot.container.load_discordbot_settings", _mock_load_settings)
    monkeypatch.setattr("clubbot.cogs.qr.load_discordbot_settings", _mock_load_settings)
    monkeypatch.setattr("clubbot.cogs.transcript.load_discordbot_settings", _mock_load_settings)

    yield


_patch_load_settings = pytest.fixture(autouse=True)(_patch_load_settings_impl)

# Convenience alias expected by some tests for direct settings construction.
_build_settings = build_settings


def pytest_sessionstart(session: pytest.Session) -> None:
    """Ensure get_event_loop() works reliably on Windows with strict asyncio mode."""
    if sys.platform.startswith("win"):
        if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
