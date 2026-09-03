"""Tests for WebSocketSniffer class and SnifferError."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.sniffer.core import (
    SnifferError,
    WebSocketSniffer,
)
from tankpit_bot.sniffer.world_service import WorldService
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


def test_websocket_sniffer_run_maximises_via_cdp_on_streamed_display(
    fake_fs: FakeFileSystem,
) -> None:
    """The streamed-display path issues Browser.setWindowBounds via CDP.

    When Vibeshine's launcher sets ``SUNSHINE_STREAM_DISPLAY_*``, the
    sniffer must (a) skip the default viewport clamp on new_context and
    (b) post-launch flip the window to the OS-maximised state via CDP.
    Exercised through the ``FakeEnv`` + fake sync-playwright pair.
    """
    from tests.conftest import FakeEnv

    _test_hooks.sync_playwright = fake_sync_playwright
    _test_hooks.get_env = FakeEnv(
        {
            "SUNSHINE_STREAM_DISPLAY_X": "0",
            "SUNSHINE_STREAM_DISPLAY_Y": "0",
            "SUNSHINE_STREAM_DISPLAY_W": "1920",
            "SUNSHINE_STREAM_DISPLAY_H": "1080",
        }
    )

    sniffer = WebSocketSniffer("https://tankpit.com")
    session = sniffer.run(1000)

    # The fake accepts the run and the sniff still captures messages;
    # if _maximize_via_cdp did not receive a ``windowId`` from the fake
    # CDP session the run would raise JSONTypeError instead of returning.
    assert session["base_url"] == "https://tankpit.com"


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
        ws = WorldService()
        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False
        sniffer._autosave_paths = ()

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

        ws = WorldService()
        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False
        sniffer._autosave_paths = ()
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

        ws = WorldService()
        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False
        sniffer._autosave_paths = ()

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

        ws = WorldService()
        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._game_log_entries = []
        sniffer._combat_tracker = None
        sniffer._live_decode = False
        sniffer._autosave_paths = ()
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
        from tankpit_bot.capture.trackers.mine import MineTracker
        from tankpit_bot.resources import static_key_file_path
        from tankpit_bot.types import CapturedMessage

        ws = WorldService()
        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(static_key_file_path(), static_key)

        # Create sniffer with minimal init
        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._live_decode = True
        sniffer._magic = "testmagic"
        sniffer._autosave_paths = ()
        sniffer._game_log_scraper = None

        # The mine tracker is the sniffer's own, not a global
        # ([[session-state-deglobalisation]] step 9).
        sniffer._mine_tracker = MineTracker()
        sniffer._mine_tracker.set_magic("testmagic")

        xor_table = sniffer._mine_tracker._xor_table
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

    def test_on_message_captured_live_decodes_received(
        self,
        fake_fs: FakeFileSystem,
    ) -> None:
        """A received frame is decoded live once the session magic is known.

        The guard is ``xor_table is not None``: frames that arrive
        before the magic cannot be decoded at all, so live decode has
        nothing to print for them. This drives the other side of it --
        table present, so the unified decoder runs and the frame lands
        in world state.
        """
        from tankpit_bot.capture.xor import build_session_xor_table
        from tankpit_bot.protocol.encoders.movement import encode_movement
        from tankpit_bot.protocol.framing import encode_frame
        from tankpit_bot.protocol.types import MovementDict
        from tankpit_bot.resources import static_key_file_path
        from tankpit_bot.state.types import make_tank_state
        from tankpit_bot.types import CapturedMessage

        ws = WorldService()
        fake_fs.write_text(static_key_file_path(), "ABCDEF" + "A" * 994)
        xor_table = build_session_xor_table("testmagic")

        # 0x47 for a non-self tank moves whichever registry tank is
        # standing on the start tile, so the registry has to know it.
        ws.world_state["tanks"] = {
            "9": make_tank_state(
                tank_id=9,
                x=50,
                y=60,
                team=1,
                rank=1,
                damage_state=3,
                name="red-9",
                is_bot=False,
                is_self=False,
                timestamp_ms=1000,
            )
        }

        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._live_decode = True
        sniffer._magic = "testmagic"
        sniffer._autosave_paths = ()
        sniffer._game_log_scraper = None
        sniffer._game_log_entries = []
        sniffer._cdp_message_buffer = []
        sniffer.xor_table = xor_table

        # 0x47 Movement: tank 9 starts at (50, 60) and walks one tile
        # east. Built with the production encoder, so a change to the
        # wire layout surfaces here instead of passing against a
        # hand-rolled shape that agrees with nothing.
        plaintext = encode_movement(
            MovementDict(
                msg_type=0x47,
                tank_id=9,
                start_x=50,
                start_y=60,
                direction=0,
                damage_state=3,
                lb_score=0,
                rank=1,
                flag=0,
                is_carrying=False,
                waypoints=[],
                path_tiles=1,
                path="e",
            )
        )
        body = bytes([0x47]) + bytes(plaintext[i] ^ xor_table[i] for i in range(len(plaintext)))
        payload = base64.b64encode(encode_frame(body)).decode()

        sniffer._on_message_captured(
            CapturedMessage(
                timestamp_ms=12345,
                direction="received",
                payload=payload,
                ws_url="wss://example.com",
            )
        )

        # The decode is the point: the walking tank reaches the registry.
        tanks = ws.get_world_state()["tanks"]
        assert [(t["tank_id"], t["x"], t["y"]) for t in tanks.values()] == [(9, 51, 60)]

    def test_autosave_capture_no_paths_is_noop(self) -> None:
        """Returns immediately when autosave is not configured."""
        ws = WorldService()
        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._autosave_paths = ()
        sniffer._target_url = "https://tankpit.com"
        sniffer._session_id = "noop"
        sniffer._start_timestamp_ms = 1000
        sniffer._messages = []
        sniffer._magic = None
        sniffer._game_log_entries = []

        sniffer._autosave_capture()

    def test_on_message_captured_autosaves_capture(self, fake_fs: FakeFileSystem) -> None:
        """Autosaves the current capture snapshot after a message arrives."""
        ws = WorldService()
        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._target_url = "https://tankpit.com"
        sniffer._headless = False
        sniffer._prefer_account = False
        sniffer._live_decode = False
        sniffer._autosave_paths = (
            Path("capture_session.json"),
            Path("runs/sniff/latest.capture_session.json"),
        )
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
        saved_latest = decode_capture_session(
            narrow_json_to_dict(
                load_json_str(fake_fs.read_text(Path("runs/sniff/latest.capture_session.json")))
            )
        )

        assert len(saved_session["messages"]) == 1
        assert saved_session["messages"][0]["payload"] == "AAAA"
        assert saved_latest == saved_session

    def test_process_game_log_entry_autosaves_game_log(self, fake_fs: FakeFileSystem) -> None:
        """Autosaves updated game log entries during capture."""
        from tankpit_bot.browser import GameLogEntry

        ws = WorldService()
        sniffer = object.__new__(WebSocketSniffer)
        sniffer.world = ws
        sniffer._cdp_service = CDPService()
        sniffer._target_url = "https://tankpit.com"
        sniffer._headless = False
        sniffer._prefer_account = False
        sniffer._live_decode = False
        sniffer._autosave_paths = (Path("capture_session.json"),)
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


def test_capture_session_carries_the_live_registry_tank_names() -> None:
    """``tank_names`` is populated from the world-state tank registry.

    Regression guard for a reader with no writer: the field used to come
    from a ``TankTracker`` that ``process_message`` was never called on,
    so it answered ``{}`` forever and all 432 archived captures carried
    an empty map. Nothing asserted the field had content, which is
    exactly why it went unnoticed ([[session-state-deglobalisation]]).
    """
    from tankpit_bot.state.types import make_tank_state

    ws = WorldService()
    service = ws
    service.world_state["tanks"] = {
        "1301": make_tank_state(
            tank_id=1301,
            x=10,
            y=20,
            team=1,
            rank=1,
            damage_state=0,
            name="Artax",
            is_bot=True,
            is_self=True,
        ),
        "500": make_tank_state(
            tank_id=500,
            x=30,
            y=40,
            team=2,
            rank=3,
            damage_state=0,
            name="red-1",
            is_bot=False,
            is_self=False,
        ),
        "501": make_tank_state(
            tank_id=501,
            x=50,
            y=60,
            team=2,
            rank=3,
            damage_state=0,
            name="",
            is_bot=False,
            is_self=False,
        ),
    }

    sniffer = object.__new__(WebSocketSniffer)
    sniffer.world = ws
    sniffer._cdp_service = CDPService()
    sniffer._session_id = "names-test"
    sniffer._start_timestamp_ms = 1000
    sniffer._target_url = "https://tankpit.com"
    sniffer._messages = []
    sniffer._ws_urls = {}
    sniffer._magic = None
    sniffer._static_key = None
    sniffer._game_log_entries = []

    session = sniffer._build_capture_session()

    # Named tanks travel with the capture; the unnamed one is omitted
    # rather than carried as an empty string.
    assert session["tank_names"] == {"1301": "Artax", "500": "red-1"}
