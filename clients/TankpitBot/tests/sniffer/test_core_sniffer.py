"""Tests for WebSocketSniffer class and SnifferError."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.sniffer.core import SnifferError, WebSocketSniffer
from tankpit_bot.types import CapturedMessage, decode_capture_session
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
    assert len(session["messages"]) == 4
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
        sniffer._cdp_service = CDPService()
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False
        sniffer._output_path = None

        entry = GameLogEntry(text="Player destroyed Enemy", category="combat")

        # Call the method directly
        sniffer._process_game_log_entry(entry)

        # Verify entry was stored
        assert len(sniffer._game_log_entries) == 1
        stored = sniffer._game_log_entries[0]
        assert stored["text"] == "Player destroyed Enemy"
        assert stored["category"] == "combat"
        assert isinstance(stored["timestamp_ms"], int) and stored["timestamp_ms"] > 0

    def test_process_game_log_entry_feeds_combat_tracker_recognized_line(self) -> None:
        """A recognized combat-category line records an event in the tracker.

        Locks the contract that the sniffer's
        ``_process_game_log_entry`` override forwards combat-category
        entries to its ``CombatTracker`` after the parent has logged
        them. The tracker is initialized via ``_init_combat_tracker``
        (sniffer-only plumbing as of 2026-06-19).
        """
        from tankpit_bot.browser import GameLogEntry

        sniffer = object.__new__(WebSocketSniffer)
        sniffer._cdp_service = CDPService()
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False
        sniffer._output_path = None
        sniffer._init_combat_tracker()

        sniffer._process_game_log_entry(
            GameLogEntry(text="You hit Tank123 for 50 damage", category="combat")
        )

        if sniffer._combat_tracker is None:
            raise AssertionError("combat tracker should be initialized")
        events = sniffer._combat_tracker.get_events()
        assert events, "combat tracker should have recorded at least one event"
        assert events[0]["attacker"] == "player"
        assert events[0]["target"] == "Tank123 for 50 damage"

    def test_process_game_log_entry_ignores_combat_when_tracker_absent(self) -> None:
        """Combat-category entries are skipped when the tracker is not initialized.

        Locks the early-return that protects the sniffer when
        ``_init_combat_tracker`` was never called -- the entry is still
        recorded and logged by the parent path, but no tracker
        interaction occurs.
        """
        from tankpit_bot.browser import GameLogEntry

        sniffer = object.__new__(WebSocketSniffer)
        sniffer._cdp_service = CDPService()
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False
        sniffer._output_path = None

        sniffer._process_game_log_entry(
            GameLogEntry(text="You hit Tank123 for 50 damage", category="combat")
        )

        assert sniffer._combat_tracker is None
        assert sniffer._game_log_entries == [
            {
                "timestamp_ms": sniffer._game_log_entries[0]["timestamp_ms"],
                "text": "You hit Tank123 for 50 damage",
                "category": "combat",
            }
        ]

    def test_process_game_log_entry_combat_tracker_ignores_unparseable_line(self) -> None:
        """Unrecognized combat text yields no tracker event but is still stored.

        Locks the second early-return in the sniffer's combat branch:
        when ``CombatTracker.process_log_line`` returns ``None`` the
        sniffer must not call ``log_event``.
        """
        from tankpit_bot.browser import GameLogEntry

        sniffer = object.__new__(WebSocketSniffer)
        sniffer._cdp_service = CDPService()
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False
        sniffer._output_path = None
        sniffer._init_combat_tracker()

        sniffer._process_game_log_entry(
            GameLogEntry(text="some unparseable combat noise xyz", category="combat")
        )

        if sniffer._combat_tracker is None:
            raise AssertionError("combat tracker should be initialized")
        assert sniffer._combat_tracker.get_events() == []
        assert sniffer._game_log_entries == [
            {
                "timestamp_ms": sniffer._game_log_entries[0]["timestamp_ms"],
                "text": "some unparseable combat noise xyz",
                "category": "combat",
            }
        ]

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
        sniffer._cdp_service = CDPService()
        sniffer._live_decode = True
        sniffer._magic = "testmagic"
        sniffer._output_path = None
        sniffer._game_log_scraper = None

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

    def test_autosave_capture_no_output_path_is_noop(self) -> None:
        """Returns immediately when autosave is not configured."""
        sniffer = object.__new__(WebSocketSniffer)
        sniffer._cdp_service = CDPService()
        sniffer._output_path = None
        sniffer._target_url = "https://tankpit.com"
        sniffer._session_id = "noop"
        sniffer._start_timestamp_ms = 1000
        sniffer._messages = []
        sniffer._magic = None
        sniffer._game_log_entries = []

        sniffer._autosave_capture()

    def test_on_message_captured_autosaves_capture(self, fake_fs: FakeFileSystem) -> None:
        """Autosaves the current capture snapshot after a message arrives."""
        sniffer = object.__new__(WebSocketSniffer)
        sniffer._cdp_service = CDPService()
        sniffer._target_url = "https://tankpit.com"
        sniffer._headless = False
        sniffer._prefer_account = False
        sniffer._live_decode = False
        sniffer._output_path = Path("capture_session.json")
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._game_log_scraper = None
        sniffer._session_id = "autosave-test"
        sniffer._start_timestamp_ms = 1000
        sniffer._cdp_message_buffer = []
        sniffer._messages = [
            CapturedMessage(
                timestamp_ms=1100,
                direction="received",
                payload="AAAA",
                ws_url="wss://example.com/ws",
            )
        ]
        sniffer._ws_urls = {}
        sniffer._magic = "magic"
        sniffer._static_key = None

        sniffer._on_message_captured(sniffer._messages[0])

        saved_session = decode_capture_session(
            narrow_json_to_dict(load_json_str(fake_fs.read_text(Path("capture_session.json"))))
        )
        saved_raw = decode_capture_session(
            narrow_json_to_dict(load_json_str(fake_fs.read_text(Path("raw_capture.json"))))
        )

        assert len(saved_session["messages"]) == 1
        assert saved_session["messages"][0]["payload"] == "AAAA"
        assert saved_raw == saved_session

    def test_process_game_log_entry_autosaves_game_log(self, fake_fs: FakeFileSystem) -> None:
        """Autosaves updated game log entries during capture."""
        from tankpit_bot.browser import GameLogEntry

        sniffer = object.__new__(WebSocketSniffer)
        sniffer._cdp_service = CDPService()
        sniffer._target_url = "https://tankpit.com"
        sniffer._headless = False
        sniffer._prefer_account = False
        sniffer._live_decode = False
        sniffer._output_path = Path("capture_session.json")
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._game_log_scraper = None
        sniffer._session_id = "autosave-log-test"
        sniffer._start_timestamp_ms = 1000
        sniffer._messages = []
        sniffer._ws_urls = {}
        sniffer._magic = None
        sniffer._static_key = None

        sniffer._process_game_log_entry(GameLogEntry(text="Zoom in", category="action"))

        saved_session = decode_capture_session(
            narrow_json_to_dict(load_json_str(fake_fs.read_text(Path("capture_session.json"))))
        )

        assert len(saved_session["game_log"]) == 1
        assert saved_session["game_log"][0]["text"] == "Zoom in"
