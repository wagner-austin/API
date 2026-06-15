"""Tests for bot main() function and error classes."""

from __future__ import annotations

from tankpit_bot.bot import BotError, ProtocolNotDiscoveredError, main


def test_main_is_entry_main() -> None:
    """Test main() is the real entry point from entry module."""
    from tankpit_bot.bot import entry

    assert main is entry.main


def test_bot_error_is_exception() -> None:
    """Test BotError is an Exception."""
    assert issubclass(BotError, Exception)
    err = BotError("test error")
    assert str(err) == "test error"


def test_protocol_not_discovered_error_is_bot_error() -> None:
    """Test ProtocolNotDiscoveredError is a BotError."""
    assert issubclass(ProtocolNotDiscoveredError, BotError)
