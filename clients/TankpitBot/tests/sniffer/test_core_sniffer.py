"""Tests for WebSocketSniffer class and SnifferError."""

from __future__ import annotations

import base64

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.sniffer import SnifferError, WebSocketSniffer
from tests.conftest import FakeFileSystem
from tests.fakes import (
    fake_sync_playwright,
    fake_sync_playwright_with_magic,
)

# =============================================================================
# WebSocketSniffer Tests
# =============================================================================


def test_websocket_sniffer_init() -> None:
    """Test WebSocketSniffer initialization."""
    sniffer = WebSocketSniffer("https://example.com", headless=True)
    assert sniffer._target_url == "https://example.com"
    assert sniffer._headless is True
    assert sniffer._live_decode is False


def test_websocket_sniffer_init_with_live_decode() -> None:
    """Test WebSocketSniffer initialization with live_decode."""
    sniffer = WebSocketSniffer("https://example.com", live_decode=True)
    assert sniffer._live_decode is True


def test_websocket_sniffer_run_without_playwright() -> None:
    """Test WebSocketSniffer.run raises error when Playwright not installed."""
    _test_hooks.sync_playwright = None
    sniffer = WebSocketSniffer("https://example.com")
    with pytest.raises(PlaywrightNotInstalledError, match="Playwright is not installed"):
        sniffer.run(1000)


def test_websocket_sniffer_run_captures_messages(fake_fs: FakeFileSystem) -> None:
    """Test WebSocketSniffer.run captures WebSocket messages."""
    _test_hooks.sync_playwright = fake_sync_playwright

    sniffer = WebSocketSniffer("https://tankpit.com", headless=True)
    session = sniffer.run(5000)

    assert session["base_url"] == "https://tankpit.com"
    # join_room has 3 wait_for_timeout calls + 1 capture wait = 4 cycles,
    # each emitting a sent+received pair = 8 messages
    assert len(session["messages"]) == 8
    assert session["messages"][0]["direction"] == "sent"
    assert session["messages"][0]["payload"] == "sent message"
    assert session["messages"][1]["direction"] == "received"
    assert session["messages"][1]["payload"] == "received message"


def test_websocket_sniffer_records_websocket_urls(fake_fs: FakeFileSystem) -> None:
    """Test WebSocketSniffer records WebSocket URLs from created events."""
    _test_hooks.sync_playwright = fake_sync_playwright

    sniffer = WebSocketSniffer("https://tankpit.com")
    session = sniffer.run(1000)

    for msg in session["messages"]:
        assert msg["ws_url"] == "wss://example.com/ws"


def test_websocket_sniffer_captures_magic(fake_fs: FakeFileSystem) -> None:
    """Test WebSocketSniffer captures tankpit.magic value."""
    _test_hooks.sync_playwright = fake_sync_playwright_with_magic

    sniffer = WebSocketSniffer("https://tankpit.com")
    session = sniffer.run(1000)

    assert session["magic"] == "test_magic_xor_key_value"


def test_websocket_sniffer_magic_none_when_not_available(fake_fs: FakeFileSystem) -> None:
    """Test WebSocketSniffer sets magic to None when tankpit.magic not available."""
    _test_hooks.sync_playwright = fake_sync_playwright

    sniffer = WebSocketSniffer("https://tankpit.com")
    session = sniffer.run(1000)

    assert session["magic"] is None


# =============================================================================
# Error Class Tests
# =============================================================================


def test_sniffer_error_is_exception() -> None:
    """Test SnifferError is an Exception."""
    assert issubclass(SnifferError, Exception)
    err = SnifferError("test error")
    assert str(err) == "test error"


# =============================================================================
# WebSocketSniffer Methods Tests
# =============================================================================


class TestWebSocketSnifferMethods:
    """Tests for WebSocketSniffer method coverage."""

    def test_process_game_log_entry_stores_entry(self) -> None:
        """Test _process_game_log_entry stores entries with timestamp."""
        from tankpit_bot.browser import GameLogEntry

        # Create sniffer with minimal init - we'll call methods directly
        # Using object.__new__ to avoid full __init__
        sniffer = object.__new__(WebSocketSniffer)
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False

        entry = GameLogEntry(text="Player destroyed Enemy", category="combat")

        # Call the method directly
        sniffer._process_game_log_entry(entry)

        # Verify entry was stored
        assert len(sniffer._game_log_entries) == 1
        stored = sniffer._game_log_entries[0]
        assert stored["text"] == "Player destroyed Enemy"
        assert stored["category"] == "combat"
        assert isinstance(stored["timestamp_ms"], int) and stored["timestamp_ms"] > 0

    def test_on_message_captured_sent_mine_status(self, fake_fs: FakeFileSystem) -> None:
        """Test _on_message_captured logs mine status for sent messages."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import trackers
        from tankpit_bot.types import CapturedMessage

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        # Reset trackers
        for tracker in trackers.ALL_TRACKERS:
            tracker._xor_table = None
            tracker._static_key = None

        # Create sniffer with minimal init
        sniffer = object.__new__(WebSocketSniffer)
        sniffer._live_decode = True
        sniffer._magic = "testmagic"
        sniffer._game_log_scraper = None
        sniffer._inventory_scraper = None

        # Initialize trackers with magic
        trackers.init_trackers_with_magic("testmagic")

        xor_table = trackers.mine_tracker._xor_table
        assert xor_table is not None and len(xor_table) == 1000

        # Create a mine drop command message (type=4, id=98)
        # Format: 0x21 '!' prefix, then XOR encoded command
        plaintext = bytes([4, 98, 50, 60])  # cmd_type=4, id=98, x=50, y=60
        body = bytes([0x21])  # Command prefix
        body += bytes(plaintext[i] ^ xor_table[i] for i in range(len(plaintext)))

        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        message = CapturedMessage(
            timestamp_ms=12345,
            direction="sent",
            payload=payload,
            ws_url="wss://example.com",
        )

        # Call _on_message_captured - this should hit the mine_status branch
        sniffer._on_message_captured(message)
