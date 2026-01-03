"""Tests for tankpit_bot.sniffer module."""

from __future__ import annotations

import base64

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.sniffer import (
    FuelTracker,
    PositionTracker,
    SnifferError,
    WebSocketSniffer,
    _decode_command,
    _decode_join_confirm,
    _decode_message,
    _decode_plus_message,
    _decode_state_message,
    _decode_text_message,
    main,
    run_sniffer,
)
from tankpit_bot.types import decode_capture_session
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes import (
    fake_sync_playwright,
    fake_sync_playwright_no_messages,
    fake_sync_playwright_with_magic,
    fake_sync_playwright_with_mixed_scripts,
    fake_sync_playwright_with_scripts,
)

# =============================================================================
# Message Decode Tests
# =============================================================================


def _make_payload(body: bytes) -> str:
    """Create a base64 payload with 2-byte length header."""
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


def test_decode_message_invalid_base64() -> None:
    """Test _decode_message handles invalid base64."""
    result = _decode_message("not valid base64!!!", "sent")
    assert result == "[SENT] (invalid base64)"


def test_decode_message_too_short() -> None:
    """Test _decode_message handles messages shorter than 2 bytes."""
    payload = base64.b64encode(b"x").decode()
    result = _decode_message(payload, "received")
    assert "[RECEIVED] (too short:" in result


def test_decode_message_auth() -> None:
    """Test _decode_message decodes AUTH messages."""
    payload = _make_payload(b"%AUTH !be 12345|token|auth extra")
    result = _decode_message(payload, "sent")
    assert result == "[SENT] AUTH: %AUTH !be 12345|token|auth extra..."


def test_decode_message_select() -> None:
    """Test _decode_message decodes SELECT messages."""
    payload = _make_payload(b"*4")
    result = _decode_message(payload, "sent")
    assert result == "[SENT] SELECT: room=4"


def test_decode_message_response() -> None:
    """Test _decode_message decodes RESPONSE messages."""
    payload = _make_payload(b"$4|0")
    result = _decode_message(payload, "received")
    assert result == "[RECEIVED] RESPONSE: $4|0"


def test_decode_message_state() -> None:
    """Test _decode_message decodes STATE messages (binary with '.' prefix)."""
    # Create a 14-byte state message (subtype 0x03, not fuel-related)
    state_body = bytes.fromhex("2e033c020300005c190000ca0300")
    payload = _make_payload(state_body)
    result = _decode_message(payload, "received")
    # 14-byte STATE message with subtype shown
    assert "[RECEIVED] STATE: sub=0x03 len=14" in result
    assert "hex=" in result


def test_decode_message_state_short() -> None:
    """Test _decode_message decodes short position messages."""
    # Short state message (4-11 bytes) - shows as POS
    short_state = bytes([0x2E, 0x01, 0x02, 0x03])  # 4 bytes
    payload = _make_payload(short_state)
    result = _decode_message(payload, "received")
    assert "[RECEIVED] POS: len=4 hex=2e010203" in result


def test_decode_text_message_state_without_body() -> None:
    """Test _decode_text_message fallback when body is None."""
    # When body is None, should fall back to simple len display
    result = _decode_text_message(".state data", 11, "RECEIVED", body=None)
    assert result == "[RECEIVED] STATE: len=11 bytes"


def test_decode_state_message_extracts_fields() -> None:
    """Test _decode_state_message handles medium-length UPDATE messages."""
    # 13-byte message falls into the 31-500 byte UPDATE category? No, actually
    # this is a short message. Let's check the length ranges in the decoder.
    # 13 bytes is in the 4-11 range for POS, but it's actually longer.
    # Actually 13 bytes is not in any category, let's use a proper test.
    # Medium messages (17-30 bytes) are ENTITY, so let's test that.
    body = bytes.fromhex("2e10200003000000190000e80300000000000000")  # 20 bytes
    result = _decode_state_message(body, "RECV")
    # 20-byte message is ENTITY type
    assert "[RECV] ENTITY: sub=0x10 len=20" in result


def test_decode_state_message_sync() -> None:
    """Test _decode_state_message handles SYNC messages (2-3 bytes)."""
    body = bytes.fromhex("2e62")  # 2 bytes
    result = _decode_state_message(body, "RECV")
    assert "[RECV] SYNC: 2e62" in result


def test_decode_state_message_map_data() -> None:
    """Test _decode_state_message handles MAP_DATA (>500 bytes)."""
    body = bytes([0x2E]) + bytes(600)  # 601 bytes total
    result = _decode_state_message(body, "RECV")
    assert "[RECV] MAP_DATA: len=601" in result


def test_decode_state_message_hit() -> None:
    """Test _decode_state_message handles HIT messages (12 bytes)."""
    body = bytes.fromhex("2e650b110f8b7bc412fd676f")  # 12 bytes
    result = _decode_state_message(body, "RECV")
    assert "[RECV] HIT: 2e650b110f8b7bc412fd676f" in result


def test_decode_state_message_entity() -> None:
    """Test _decode_state_message handles ENTITY messages (17-30 bytes)."""
    body = bytes([0x2E]) + bytes(19)  # 20 bytes total, subtype is 0x00
    result = _decode_state_message(body, "RECV")
    assert "[RECV] ENTITY: sub=0x00 len=20" in result


def test_decode_state_message_update() -> None:
    """Test _decode_state_message handles UPDATE messages (31-500 bytes)."""
    body = bytes([0x2E]) + bytes(49)  # 50 bytes total
    result = _decode_state_message(body, "RECV")
    assert "[RECV] UPDATE: len=50" in result


def test_decode_message_unknown() -> None:
    """Test _decode_message handles unknown message types."""
    payload = _make_payload(b"some unknown message format")
    result = _decode_message(payload, "received")
    assert "[RECEIVED] ???:" in result


def test_decode_message_quit() -> None:
    """Test _decode_message decodes QUIT messages (dash character)."""
    payload = _make_payload(b"-")
    result = _decode_message(payload, "sent")
    assert result == "[SENT] QUIT: -"


def test_decode_plus_message_room_list() -> None:
    """Test _decode_plus_message decodes ROOM_LIST messages."""
    result = _decode_plus_message("+4|World (Meltdown)|42|flags", "RECV")
    assert result == "[RECV] ROOM_LIST: room=4 name=World (Meltdown)"


def test_decode_plus_message_action() -> None:
    """Test _decode_plus_message decodes ACTION messages (non-room-list format)."""
    # Room list format has name as 2nd field, action has numeric 2nd field
    # But our detector checks if 3rd field exists and 1st field after + is digit
    # So we need a case where it's NOT a room list - when parts[0][1:] is not all digits
    result = _decode_plus_message("+x|2|116|79|extra", "SENT")
    assert result == "[SENT] ACTION: room=x coords=116,79"


def test_decode_plus_message_action_short() -> None:
    """Test _decode_plus_message handles short ACTION messages."""
    result = _decode_plus_message("+4|2", "SENT")
    assert result == "[SENT] ACTION: room=4 coords=?"


def test_decode_join_confirm() -> None:
    """Test _decode_join_confirm decodes JOIN_CONFIRM messages."""
    result = _decode_join_confirm("=4|Sep. 25, 2012|Yuppler|4|9|10", "RECV")
    assert result == "[RECV] JOIN_CONFIRM: room=4 tank=Yuppler"


def test_decode_join_confirm_short() -> None:
    """Test _decode_join_confirm handles short messages."""
    result = _decode_join_confirm("=4|date", "RECV")
    assert result == "[RECV] JOIN_CONFIRM: room=4 tank=?"


def test_decode_command_with_rest() -> None:
    """Test _decode_command decodes commands with additional data (no magic)."""
    # Use actual binary bytes, not ASCII string
    body = bytes([0x21, 0x31, 0x2D, 0x43, 0xFE])  # !1-C<0xFE>
    result = _decode_command(body, "SENT")
    assert result == "[SENT] CMD: ! 21312d43fe"


def test_decode_command_with_magic_xor_decryption() -> None:
    """Test _decode_command decodes commands with XOR decryption when magic provided."""
    from pathlib import Path

    # Check if static key exists (required for XOR decryption)
    static_key_path = Path(__file__).parent.parent / "xor_static_key.txt"
    if not static_key_path.exists():
        pytest.skip("xor_static_key.txt not found")

    # Read the static key
    static_key = static_key_path.read_text().strip()
    magic = "test_magic_key_20char"  # 20 char magic key

    # Build the XOR table manually to know what encoded bytes to send
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])

    # We want to decode type=2, id=63 (enter game command)
    # Encoded bytes: type_encoded = 2 ^ table[0], id_encoded = 63 ^ table[1]
    type_encoded = 2 ^ table[0]
    id_encoded = 63 ^ table[1]
    body = bytes([0x21, type_encoded, id_encoded])

    result = _decode_command(body, "SENT", magic)
    assert result == "[SENT] CMD: ! type=2 id=63"


def test_decode_command_short() -> None:
    """Test _decode_command handles short command messages."""
    result = _decode_command(b"!", "SENT")
    assert result == "[SENT] CMD: ! (too short: 21)"


def test_decode_command_non_ascii() -> None:
    """Test _decode_command handles non-ASCII command bytes (no magic)."""
    body = bytes([0x21, 0x90, 0xAB, 0xCD])  # !<0x90><0xAB><0xCD>
    result = _decode_command(body, "SENT")
    assert result == "[SENT] CMD: ! 2190abcd"


def test_decode_message_calls_decode_plus_for_room_list() -> None:
    """Test _decode_message routes to _decode_plus_message for ROOM_LIST."""
    payload = _make_payload(b"+3|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2025")
    result = _decode_message(payload, "received")
    assert result == "[RECEIVED] ROOM_LIST: room=3 name=Practice"


def test_decode_message_calls_decode_join_confirm() -> None:
    """Test _decode_message routes to _decode_join_confirm."""
    payload = _make_payload(b"=4|Sep. 25, 2012|Yuppler|4|9|10|10|9")
    result = _decode_message(payload, "received")
    assert result == "[RECEIVED] JOIN_CONFIRM: room=4 tank=Yuppler"


def test_decode_message_calls_decode_command() -> None:
    """Test _decode_message routes to _decode_command."""
    payload = _make_payload(b"!7b")
    result = _decode_message(payload, "sent")
    # Without magic key, just shows raw hex
    assert result == "[SENT] CMD: ! 213762"


def test_decode_message_command_with_magic_but_no_static_key() -> None:
    """Test _decode_message falls back to hex when static key file doesn't exist.

    This covers the branch where magic is provided but static key file is missing.
    """
    from tests.conftest import FakeFileSystem

    # Create a fake filesystem that has NO static key file
    fs = FakeFileSystem()
    _test_hooks.read_text = fs.read_text
    _test_hooks.path_exists = fs.path_exists

    payload = _make_payload(b"!7b")
    # Provide magic key, but static key file doesn't exist
    result = _decode_message(payload, "sent", magic="test_magic")

    # Should fall back to hex output since static key file is missing
    assert result == "[SENT] CMD: ! 213762"


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
    assert len(session["messages"]) == 2
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
# run_sniffer Tests
# =============================================================================


def test_run_sniffer_saves_to_file(fake_fs: FakeFileSystem) -> None:
    """Test run_sniffer saves capture session to file."""
    _test_hooks.sync_playwright = fake_sync_playwright

    session = run_sniffer(
        "https://tankpit.com",
        "output.json",
        headless=True,
        capture_duration_ms=1000,
    )

    written_files = fake_fs.get_written_files()
    content = written_files["output.json"]
    parsed = load_json_str(content)
    parsed_dict = narrow_json_to_dict(parsed)
    decoded = decode_capture_session(parsed_dict)
    assert decoded["session_id"] == session["session_id"]


def test_run_sniffer_with_live_decode(fake_fs: FakeFileSystem) -> None:
    """Test run_sniffer with live_decode enabled."""
    _test_hooks.sync_playwright = fake_sync_playwright

    session = run_sniffer(
        "https://tankpit.com",
        "output.json",
        live_decode=True,
    )

    assert len(session["messages"]) == 2


# =============================================================================
# main() Tests
# =============================================================================


def test_main_with_defaults(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() uses default values when env vars not set."""
    _test_hooks.sync_playwright = fake_sync_playwright

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Captured 2 WebSocket messages in" in output
    assert "Saved to: capture_session.json" in output


def test_main_with_custom_env(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() reads custom values from environment."""
    _test_hooks.sync_playwright = fake_sync_playwright
    fake_env.set("TANKPIT_URL", "https://custom.tankpit.com")
    fake_env.set("TANKPIT_OUTPUT", "custom_output.json")
    fake_env.set("TANKPIT_HEADLESS", "true")
    fake_env.set("TANKPIT_DURATION_MS", "5000")

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Saved to: custom_output.json" in output


def test_main_headless_variations(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test main() parses various headless env values."""
    _test_hooks.sync_playwright = fake_sync_playwright

    fake_env.set("TANKPIT_HEADLESS", "1")
    main()

    fake_env.set("TANKPIT_HEADLESS", "yes")
    main()

    fake_env.set("TANKPIT_HEADLESS", "TRUE")
    main()


def test_main_live_decode_disabled(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test main() can disable live decode via env var."""
    _test_hooks.sync_playwright = fake_sync_playwright

    fake_env.set("TANKPIT_LIVE_DECODE", "false")
    main()

    fake_env.set("TANKPIT_LIVE_DECODE", "0")
    main()

    fake_env.set("TANKPIT_LIVE_DECODE", "no")
    main()


def test_main_prints_discovered_urls(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints discovered WebSocket URLs."""
    _test_hooks.sync_playwright = fake_sync_playwright

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Discovered WebSocket URLs (1):" in output
    assert "wss://example.com/ws" in output


def test_main_installs_playwright_when_none(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test main() installs playwright via get_sync_playwright when None."""

    def get_fake_factory() -> SyncPlaywrightFactoryProtocol:
        """Return the fake sync_playwright factory function."""
        return fake_sync_playwright

    _test_hooks.sync_playwright = None
    _test_hooks.get_sync_playwright = get_fake_factory

    main()

    assert _test_hooks.sync_playwright == fake_sync_playwright


def test_main_with_no_websocket_urls(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() when no WebSocket URLs are discovered."""
    _test_hooks.sync_playwright = fake_sync_playwright_no_messages

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Captured 0 WebSocket messages in" in output
    assert "Discovered WebSocket URLs" not in output


def test_main_logs_script_urls(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() logs script URLs discovered on the page."""
    _test_hooks.sync_playwright = fake_sync_playwright_with_scripts

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Loaded scripts (2):" in output
    assert "- https://tankpit.com/js/game.js" in output
    assert "- https://tankpit.com/js/protocol.js" in output


def test_main_logs_only_string_script_urls(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() only logs script URLs that are strings, skipping non-strings."""
    _test_hooks.sync_playwright = fake_sync_playwright_with_mixed_scripts

    main()

    captured = capsys.readouterr()
    output = captured.out
    # Should log the valid string URLs
    assert "Loaded scripts (4):" in output
    assert "- https://tankpit.com/js/valid.js" in output
    assert "- https://tankpit.com/js/another.js" in output
    # Should NOT log the non-string values (123, None)
    assert "- 123" not in output
    assert "- None" not in output


# =============================================================================
# Error Class Tests
# =============================================================================


def test_sniffer_error_is_exception() -> None:
    """Test SnifferError is an Exception."""
    assert issubclass(SnifferError, Exception)
    err = SnifferError("test error")
    assert str(err) == "test error"


# =============================================================================
# FuelTracker Tests
# =============================================================================


def test_fuel_tracker_set_magic_builds_xor_table() -> None:
    """Test FuelTracker.set_magic builds XOR table from static key."""
    tracker = FuelTracker()
    assert tracker._xor_table is None

    # Use known magic from captured session
    tracker.set_magic("bos3209i3e12bojm6wn9")

    assert tracker._xor_table is not None
    # XOR table is built from full 1000-char static key
    assert len(tracker._xor_table) == 1000


def test_fuel_tracker_decode_fuel_u16() -> None:
    """Test FuelTracker.decode_fuel returns u16 value."""
    tracker = FuelTracker()
    tracker.set_magic("bos3209i3e12bojm6wn9")

    # From session 3: 2e155d0b526b4b7069470268344a -> u16 = 17274
    body = bytes.fromhex("2e155d0b526b4b7069470268344a")
    fuel = tracker.decode_fuel(body)
    assert fuel == 17274


def test_fuel_tracker_decode_fuel_radar_delta() -> None:
    """Test fuel decrements by 10 after radar use."""
    tracker = FuelTracker()
    tracker.set_magic("bos3209i3e12bojm6wn9")

    # Before radar: u16 = 17322
    body1 = bytes.fromhex("2e155d0b526b4b7069470268e44a")
    fuel1 = tracker.decode_fuel(body1)

    # After radar: u16 = 17312 (-10)
    body2 = bytes.fromhex("2e155d0b526b4b7069470268ee4a")
    fuel2 = tracker.decode_fuel(body2)

    assert fuel1 - fuel2 == 10  # Radar costs 10 fuel


def test_fuel_tracker_decode_fuel_deposit_delta() -> None:
    """Test fuel decrements by 100 after deposit."""
    tracker = FuelTracker()
    tracker.set_magic("bos3209i3e12bojm6wn9")

    # Before deposit: u16 = 17274 (raw bytes 0x34, 0x4a at pos 12-13)
    body1 = bytes.fromhex("2e155d0b526b4b7069470268344a")
    fuel1 = tracker.decode_fuel(body1)

    # After deposit 100: u16 = 17174 (raw bytes 0x58, 0x4a at pos 12-13)
    body2 = bytes.fromhex("2e155d0b526b4b7069470268584a")
    fuel2 = tracker.decode_fuel(body2)

    assert fuel1 - fuel2 == 100  # Deposit costs 100 fuel


def test_fuel_tracker_decode_fuel_wrong_length_returns_none() -> None:
    """Test decode_fuel returns None for wrong length."""
    tracker = FuelTracker()
    tracker.set_magic("bos3209i3e12bojm6wn9")

    # Too short (12 bytes instead of 14)
    body = bytes.fromhex("2e155d0b526b4b706947")
    fuel = tracker.decode_fuel(body)
    assert fuel is None


def test_fuel_tracker_decode_fuel_no_magic_returns_none() -> None:
    """Test decode_fuel returns None when magic not set."""
    tracker = FuelTracker()
    # Don't set magic

    body = bytes.fromhex("2e155d0b526b4b7069470268344a")
    fuel = tracker.decode_fuel(body)
    assert fuel is None


def test_fuel_tracker_process_message_14byte() -> None:
    """Test process_message handles 14-byte fuel messages."""
    tracker = FuelTracker()
    tracker.set_magic("bos3209i3e12bojm6wn9")

    # Create payload: 2-byte header + 14-byte body
    body = bytes.fromhex("2e155d0b526b4b7069470268344a")
    header = len(body).to_bytes(2, "little")
    payload = base64.b64encode(header + body).decode()

    result = tracker.process_message(payload)
    assert result is not None
    assert "[FUEL:0x15]" in result
    assert "17274" in result


def test_fuel_tracker_process_message_shows_radar_delta() -> None:
    """Test process_message shows [radar] tag for -10 delta."""
    tracker = FuelTracker()
    tracker.set_magic("bos3209i3e12bojm6wn9")

    # First message: u16 = 17322
    body1 = bytes.fromhex("2e155d0b526b4b7069470268e44a")
    header1 = len(body1).to_bytes(2, "little")
    payload1 = base64.b64encode(header1 + body1).decode()
    tracker.process_message(payload1)

    # Second message: u16 = 17312 (-10)
    body2 = bytes.fromhex("2e155d0b526b4b7069470268ee4a")
    header2 = len(body2).to_bytes(2, "little")
    payload2 = base64.b64encode(header2 + body2).decode()
    result = tracker.process_message(payload2)

    assert result is not None
    assert "(-10)" in result
    assert "[radar]" in result


def test_fuel_tracker_process_message_invalid_base64_returns_none() -> None:
    """Test process_message returns None for invalid base64."""
    tracker = FuelTracker()
    tracker.set_magic("752dul1q9avle8ot5jh4")

    result = tracker.process_message("not-valid-base64!!!")
    assert result is None


# =============================================================================
# PositionTracker Tests
# =============================================================================


def test_position_tracker_set_magic_builds_xor_table() -> None:
    """Test PositionTracker.set_magic builds XOR table from static key."""
    tracker = PositionTracker()
    assert tracker._xor_table is None

    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    assert tracker._xor_table is not None
    assert len(tracker._xor_table) == 1000


def test_position_tracker_decode_position_from_0x75() -> None:
    """Test PositionTracker.decode_position extracts x,y from movement response."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After MOVE TO (93, 113): 0x75 shows FROM position (93, 118)
    # raw=2e757d7e584a0932765910304b1f690d473d20 (19 bytes)
    body = bytes.fromhex("2e757d7e584a0932765910304b1f690d473d20")
    pos = tracker.decode_position(body)
    assert pos == (93, 118)


def test_position_tracker_decode_position_after_move() -> None:
    """Test position decoding shows previous position after movement."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After MOVE (93,113)->(93,118): 0x75 shows (93, 113) = FROM position
    # raw=2e757d7e584d0132765910304b1f74105a203d
    body = bytes.fromhex("2e757d7e584d0132765910304b1f74105a203d")
    pos = tracker.decode_position(body)
    assert pos == (93, 113)


def test_position_tracker_decode_position_x_changes() -> None:
    """Test position decoding when x coordinate changes."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After MOVE (90,118)->(93,118): 0x75 shows (90, 118) = FROM position
    # raw=2e757d7e5f4a0d32765910304b1f62064c
    body = bytes.fromhex("2e757d7e5f4a0d32765910304b1f62064c")
    pos = tracker.decode_position(body)
    assert pos == (90, 118)


def test_position_tracker_decode_position_diagonal() -> None:
    """Test position decoding after diagonal movement."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After MOVE (91,115)->(93,118): 0x75 shows (91, 115) = FROM position
    # raw=2e757d7e5e4f0f32765910304b1f74105a362b
    body = bytes.fromhex("2e757d7e5e4f0f32765910304b1f74105a362b")
    pos = tracker.decode_position(body)
    assert pos == (91, 115)


def test_position_tracker_decode_position_wrong_type_returns_none() -> None:
    """Test decode_position returns None for non-0x75 messages."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # 0x1c message (fuel), not 0x75
    body = bytes.fromhex("2e1c4240073f033137191232741b")
    pos = tracker.decode_position(body)
    assert pos is None


def test_position_tracker_decode_position_no_magic_returns_none() -> None:
    """Test decode_position returns None without magic key."""
    tracker = PositionTracker()
    # No magic set

    body = bytes.fromhex("2e757d7e584a0932765910304b1f690d473d20")
    pos = tracker.decode_position(body)
    assert pos is None


def test_position_tracker_process_message_returns_status() -> None:
    """Test process_message returns formatted position status."""
    tracker = PositionTracker()
    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # Build payload with 2-byte length header
    body = bytes.fromhex("2e757d7e584a0932765910304b1f690d473d20")
    length = len(body)
    payload_bytes = length.to_bytes(2, "little") + body
    payload = base64.b64encode(payload_bytes).decode()

    result = tracker.process_message(payload)
    assert result is not None
    assert "[POS:FROM]" in result
    assert "(93, 118)" in result


def test_position_tracker_update_from_move() -> None:
    """Test update_from_move sets current position."""
    tracker = PositionTracker()
    assert tracker.current_position is None

    tracker.update_from_move(93, 118)
    assert tracker.current_position == (93, 118)

    tracker.update_from_move(90, 115)
    assert tracker.current_position == (90, 115)


def test_position_tracker_is_blocked_response() -> None:
    """Test is_blocked_response detects 5-byte blocking messages."""
    tracker = PositionTracker()

    # Blocked movement response (5 bytes)
    blocked = bytes.fromhex("2e6347320d")
    assert tracker.is_blocked_response(blocked) is True

    # Normal movement response (not blocked)
    normal = bytes.fromhex("2e757d7e584a0932765910304b1f690d473d20")
    assert tracker.is_blocked_response(normal) is False

    # Too short
    short = bytes.fromhex("2e63")
    assert tracker.is_blocked_response(short) is False


def test_position_tracker_process_message_blocked() -> None:
    """Test process_message returns BLOCKED status for 5-byte messages."""
    tracker = PositionTracker()
    tracker.set_magic("hwvoiew1x26uiv6zlvas")

    # Build payload with 2-byte length header
    body = bytes.fromhex("2e6347320d")
    length = len(body)
    payload_bytes = length.to_bytes(2, "little") + body
    payload = base64.b64encode(payload_bytes).decode()

    result = tracker.process_message(payload)
    assert result == "[POS:BLOCKED]"


def test_position_tracker_variable_subtype() -> None:
    """Test position decoding works with different subtypes per session."""
    tracker = PositionTracker()
    tracker.set_magic("hwvoiew1x26uiv6zlvas")

    # Session with 0x76 subtype (different from 0x75)
    # After MOVE: 0x76 len=20, raw=2e767a306a4e1c3d704c576d084575004f3b3f6e
    body = bytes.fromhex("2e767a306a4e1c3d704c576d084575004f3b3f6e")
    pos = tracker.decode_position(body)
    # Should decode position regardless of subtype
    assert pos is not None
    # Verify subtype was tracked
    assert tracker._move_subtype == 0x76
