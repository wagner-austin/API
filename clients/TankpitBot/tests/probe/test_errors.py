"""Tests for probe error classes."""

from __future__ import annotations

from tankpit_bot.browser import BrowserError
from tankpit_bot.probe import (
    GameNotJoinedError,
    PlaywrightNotInstalledError,
    ProbeError,
)


def test_probe_error_is_exception() -> None:
    """Test ProbeError is an Exception."""
    assert issubclass(ProbeError, Exception)
    err = ProbeError("test error")
    assert str(err) == "test error"


def test_playwright_not_installed_error_is_browser_error() -> None:
    """Test PlaywrightNotInstalledError is a BrowserError."""
    assert issubclass(PlaywrightNotInstalledError, BrowserError)


def test_game_not_joined_error_is_browser_error() -> None:
    """Test GameNotJoinedError is a BrowserError."""
    assert issubclass(GameNotJoinedError, BrowserError)
