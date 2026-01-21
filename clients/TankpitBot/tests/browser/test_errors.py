"""Tests for browser error classes."""

from __future__ import annotations

from tankpit_bot.browser import (
    BrowserError,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
)


def test_browser_error_is_exception() -> None:
    """Test BrowserError is a subclass of Exception."""
    assert issubclass(BrowserError, Exception)
    err = BrowserError("test error")
    assert str(err) == "test error"


def test_playwright_not_installed_error_is_browser_error() -> None:
    """Test PlaywrightNotInstalledError is a BrowserError."""
    assert issubclass(PlaywrightNotInstalledError, BrowserError)


def test_game_not_joined_error_is_browser_error() -> None:
    """Test GameNotJoinedError is a BrowserError."""
    assert issubclass(GameNotJoinedError, BrowserError)
