"""Tests for tankpit_bot.sniffer module."""

from __future__ import annotations

import base64

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.sniffer import (
    PositionTracker,
    SnifferError,
    WebSocketSniffer,
    _decode_command,
    _decode_join_confirm,
    _decode_message,
    _decode_plus_message,
    _decode_state_message,
    _decode_text_message,
    _identify_by_length,
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
    assert result == "[RECV] JOIN_CONFIRM: room=4 tank=Yuppler lieutenant"


def test_decode_join_confirm_short() -> None:
    """Test _decode_join_confirm handles short messages."""
    result = _decode_join_confirm("=4|date", "RECV")
    assert result == "[RECV] JOIN_CONFIRM: room=4 tank=? rank-1"


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
    assert result == "[RECEIVED] JOIN_CONFIRM: room=4 tank=Yuppler lieutenant"


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
# PositionTracker Tests
# =============================================================================


def test_position_tracker_set_magic_builds_xor_table() -> None:
    """Test PositionTracker.set_magic builds XOR table from static key."""
    tracker = PositionTracker()
    assert tracker._xor_table is None

    tracker.set_magic("kp8ffxx7muk63a0ywtqh")

    # After set_magic, _xor_table must be populated with 1000 bytes
    xor_table = tracker._xor_table
    if xor_table is None:
        raise AssertionError("_xor_table was not populated after set_magic")
    assert len(xor_table) == 1000


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
    if result is None:
        raise AssertionError("process_message returned None for valid payload")
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
    assert pos == (102, 125)
    # Verify subtype was tracked
    assert tracker._move_subtype == 0x76


# =============================================================================
# _identify_by_length Tests
# =============================================================================


class TestIdentifyByLength:
    """Tests for _identify_by_length function."""

    def test_heartbeat_1_byte(self) -> None:
        """Test 1-byte messages identified as heartbeat."""
        result = _identify_by_length(b"\x5f")
        assert result == ("heartbeat", "IDENTIFIED")

    def test_tank_status_sync_2_bytes(self) -> None:
        """Test 2-byte messages identified as tank_status_sync."""
        result = _identify_by_length(b"\x2e\x62")
        assert result == ("tank_status_sync", "FULL")

    def test_tank_status_sync_3_bytes(self) -> None:
        """Test 3-byte messages identified as tank_status_sync."""
        result = _identify_by_length(b"\x2e\x62\x01")
        assert result == ("tank_status_sync", "FULL")

    def test_entity_sync_5_bytes(self) -> None:
        """Test 5-byte messages identified as entity_sync."""
        result = _identify_by_length(bytes(5))
        assert result == ("entity_sync", "IDENTIFIED")

    def test_control_msg_6_bytes(self) -> None:
        """Test 6-byte messages identified as control_msg."""
        result = _identify_by_length(bytes(6))
        assert result == ("control_msg", "IDENTIFIED")

    def test_action_ack_7_bytes(self) -> None:
        """Test 7-byte messages identified as action_ack."""
        result = _identify_by_length(bytes(7))
        assert result == ("action_ack", "IDENTIFIED")

    def test_tank_status_short_9_bytes(self) -> None:
        """Test 9-byte messages identified as tank_status_short."""
        result = _identify_by_length(bytes(9))
        assert result == ("tank_status_short", "FULL")

    def test_tank_update_compact_10_bytes(self) -> None:
        """Test 10-byte messages identified as tank_update_compact."""
        result = _identify_by_length(bytes(10))
        assert result == ("tank_update_compact", "FULL")

    def test_tank_update_extended_14_bytes(self) -> None:
        """Test 14-byte messages identified as tank_update_extended."""
        result = _identify_by_length(bytes(14))
        assert result == ("tank_update_extended", "FULL")

    def test_tank_update_full_15_bytes(self) -> None:
        """Test 15-byte messages identified as tank_update_full."""
        result = _identify_by_length(bytes(15))
        assert result == ("tank_update_full", "FULL")

    def test_combat_hit_11_bytes(self) -> None:
        """Test 11-byte messages identified as combat_hit."""
        result = _identify_by_length(bytes(11))
        assert result == ("combat_hit", "FULL")

    def test_position_update_13_bytes(self) -> None:
        """Test 13-byte messages identified as position_update."""
        result = _identify_by_length(bytes(13))
        assert result == ("position_update", "FULL")

    def test_tank_registry_16_to_20_bytes(self) -> None:
        """Test 16-20 byte messages identified as tank_registry."""
        for length in (16, 17, 18, 19, 20):
            result = _identify_by_length(bytes(length))
            assert result == ("tank_registry", "FULL"), f"Failed for length {length}"

    def test_entity_extended_21_to_28_bytes(self) -> None:
        """Test 21-28 byte messages identified as entity_extended."""
        for length in (21, 24, 28):
            result = _identify_by_length(bytes(length))
            assert result == ("entity_extended", "IDENTIFIED"), f"Failed for length {length}"

    def test_player_ack_4_bytes(self) -> None:
        """Test 4 byte messages identified as player_ack."""
        result = _identify_by_length(bytes(4))
        assert result == ("player_ack", "IDENTIFIED")

    def test_tip_notification_29_to_60_bytes(self) -> None:
        """Test 29-60 byte messages identified as tip_notification."""
        for length in (29, 35, 47, 55, 60):
            result = _identify_by_length(bytes(length))
            assert result == ("tip_notification", "IDENTIFIED"), f"Failed for length {length}"

    def test_chunk_data_80_to_130_bytes(self) -> None:
        """Test 80-130 byte messages identified as chunk_data."""
        for length in (80, 83, 127, 130):
            result = _identify_by_length(bytes(length))
            assert result == ("chunk_data", "IDENTIFIED"), f"Failed for length {length}"

    def test_world_state_500_plus_bytes(self) -> None:
        """Test 500+ byte messages identified as world_state."""
        for length in (500, 617, 623, 627, 1000):
            result = _identify_by_length(bytes(length))
            assert result == ("world_state", "IDENTIFIED"), f"Failed for length {length}"

    def test_unknown_length_returns_none(self) -> None:
        """Test unrecognized lengths return None."""
        # Gaps in the length patterns (4 is player_ack, 29-60 is tip_notification)
        unknown_lengths = [8, 12, 61, 79, 131, 200, 400]
        for length in unknown_lengths:
            result = _identify_by_length(bytes(length))
            assert result is None, f"Expected None for length {length}, got {result}"

    def test_empty_data_returns_none(self) -> None:
        """Test empty data returns None."""
        result = _identify_by_length(b"")
        assert result is None


# =============================================================================
# DeactivationTracker Tests
# =============================================================================


class TestDeactivationTracker:
    """Tests for DeactivationTracker class."""

    def test_init(self) -> None:
        """Test DeactivationTracker initialization."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._my_tank_id is None
        assert tracker._kills == 0
        assert tracker._deaths == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_set_my_tank_id(self) -> None:
        """Test set_my_tank_id stores tank ID."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        tracker.set_my_tank_id(123)
        assert tracker._my_tank_id == 123

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_non_0x2e(self) -> None:
        """Test process_message returns None for non-0x2E messages."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = _make_payload(b"\x30\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_wrong_length(self) -> None:
        """Test process_message returns None for wrong length messages."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        # 7 bytes instead of 8
        payload = _make_payload(b"\x2e\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_kills_property(self) -> None:
        """Test kills property returns kill count."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        assert tracker.kills == 0

    def test_deaths_property(self) -> None:
        """Test deaths property returns death count."""
        from tankpit_bot.sniffer import DeactivationTracker

        tracker = DeactivationTracker()
        assert tracker.deaths == 0


# =============================================================================
# ItemPickupTracker Tests
# =============================================================================


class TestItemPickupTracker:
    """Tests for ItemPickupTracker class."""

    def test_init(self) -> None:
        """Test ItemPickupTracker initialization."""
        from tankpit_bot.sniffer import ItemPickupTracker

        tracker = ItemPickupTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._total_armor == 0
        assert tracker._total_missile == 0
        assert tracker._total_homing == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import ItemPickupTracker

        tracker = ItemPickupTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import ItemPickupTracker

        tracker = ItemPickupTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import ItemPickupTracker

        tracker = ItemPickupTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data."""
        from tankpit_bot.sniffer import ItemPickupTracker

        tracker = ItemPickupTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_non_0x2e(self) -> None:
        """Test process_message returns None for non-0x2E messages."""
        from tankpit_bot.sniffer import ItemPickupTracker

        tracker = ItemPickupTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = _make_payload(b"\x30\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None


# =============================================================================
# RadarTracker Tests
# =============================================================================


class TestRadarTracker:
    """Tests for RadarTracker class."""

    def test_init(self) -> None:
        """Test RadarTracker initialization."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_classify_entity_tank(self) -> None:
        """Test _classify_entity identifies tanks (0xFFFF)."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(10, 20, 0xFFFF)
        assert category == "tanks"
        assert formatted == "(10,20)"

    def test_classify_entity_equipment(self) -> None:
        """Test _classify_entity identifies equipment (>= 0x8000)."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(15, 25, 0x8005)
        assert category == "equip"
        assert "(15,25)" in formatted

    def test_classify_entity_fuel(self) -> None:
        """Test _classify_entity identifies fuel (< 0x8000)."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(5, 10, 500)
        assert category == "fuel"
        assert formatted == "(5,10)=500"

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        payload = _make_payload(b"\x2e\x70\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


# =============================================================================
# TankTracker Tests
# =============================================================================


class TestTankTracker:
    """Tests for TankTracker class."""

    def test_init(self) -> None:
        """Test TankTracker initialization."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._tanks == {}

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_register_name(self) -> None:
        """Test register_name stores tank name."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        tracker.register_name(123, "TestTank")

        assert tracker._tanks[123]["name"] == "TestTank"

    def test_register_name_updates_existing(self) -> None:
        """Test register_name updates existing tank entry."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        tracker._tanks[123] = {"team": "red", "rank": "private"}
        tracker.register_name(123, "NewName")

        assert tracker._tanks[123]["name"] == "NewName"
        assert tracker._tanks[123]["team"] == "red"

    def test_get_name_returns_name(self) -> None:
        """Test get_name returns stored name."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        tracker.register_name(123, "TestTank")

        result = tracker.get_name(123)
        assert result == "TestTank"

    def test_get_name_returns_none_for_unknown(self) -> None:
        """Test get_name returns None for unknown tank."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        result = tracker.get_name(999)
        assert result is None

    def test_get_name_returns_none_if_name_not_set(self) -> None:
        """Test get_name returns None if name not set."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        tracker._tanks[123] = {"team": "red"}

        result = tracker.get_name(123)
        assert result is None

    def test_get_all_names_returns_dict(self) -> None:
        """Test get_all_names returns all name mappings."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        tracker.register_name(1, "Tank1")
        tracker.register_name(2, "Tank2")
        tracker._tanks[3] = {"team": "blue"}  # No name

        result = tracker.get_all_names()
        assert result == {1: "Tank1", 2: "Tank2"}

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


# =============================================================================
# MineTracker Tests
# =============================================================================


class TestMineTracker:
    """Tests for MineTracker class."""

    def test_init(self) -> None:
        """Test MineTracker initialization."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._mines_placed == 0
        assert tracker._mines_detonated == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_mines_placed_property(self) -> None:
        """Test mines_placed property returns count."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        assert tracker.mines_placed == 0

    def test_mines_detonated_property(self) -> None:
        """Test mines_detonated property returns count."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        assert tracker.mines_detonated == 0

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        payload = _make_payload(b"\x2e\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None


# =============================================================================
# EquipmentToggleTracker Tests
# =============================================================================


class TestEquipmentToggleTracker:
    """Tests for EquipmentToggleTracker class."""

    def test_init(self) -> None:
        """Test EquipmentToggleTracker initialization."""
        from tankpit_bot.sniffer import EquipmentToggleTracker

        tracker = EquipmentToggleTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._state == [False, False, False, False, False]
        assert tracker._prev_state is None

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import EquipmentToggleTracker

        tracker = EquipmentToggleTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_state_property(self) -> None:
        """Test state property returns equipment state dict."""
        from tankpit_bot.sniffer import EquipmentToggleTracker

        tracker = EquipmentToggleTracker()
        state = tracker.state
        assert state == {
            "armor": False,
            "dual": False,
            "missile": False,
            "homing": False,
            "radar": False,
        }

    def test_detect_changes_no_previous(self) -> None:
        """Test _detect_changes returns empty list without previous state."""
        from tankpit_bot.sniffer import EquipmentToggleTracker

        tracker = EquipmentToggleTracker()
        changes = tracker._detect_changes([True, False, True, False, True])
        assert changes == []

    def test_detect_changes_with_changes(self) -> None:
        """Test _detect_changes detects state changes."""
        from tankpit_bot.sniffer import EquipmentToggleTracker

        tracker = EquipmentToggleTracker()
        tracker._prev_state = [False, False, False, False, False]
        changes = tracker._detect_changes([True, False, True, False, False])
        assert "armor=ON" in changes
        assert "missile=ON" in changes

    def test_detect_changes_off_transitions(self) -> None:
        """Test _detect_changes detects OFF transitions."""
        from tankpit_bot.sniffer import EquipmentToggleTracker

        tracker = EquipmentToggleTracker()
        tracker._prev_state = [True, True, False, False, False]
        changes = tracker._detect_changes([False, True, False, False, False])
        assert "armor=OFF" in changes
        assert len(changes) == 1

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import EquipmentToggleTracker

        tracker = EquipmentToggleTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import EquipmentToggleTracker

        tracker = EquipmentToggleTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


# =============================================================================
# ContainerTracker Tests
# =============================================================================


class TestContainerTracker:
    """Tests for ContainerTracker class."""

    def test_init(self) -> None:
        """Test ContainerTracker initialization."""
        from tankpit_bot.sniffer import ContainerTracker

        tracker = ContainerTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._containers == {}

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import ContainerTracker

        tracker = ContainerTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_containers_property(self) -> None:
        """Test containers property returns copy of container dict."""
        from tankpit_bot.sniffer import ContainerTracker

        tracker = ContainerTracker()
        tracker._containers = {1: 100, 2: 200}
        result = tracker.containers
        assert result == {1: 100, 2: 200}
        # Verify it's a copy
        result[3] = 300
        assert 3 not in tracker._containers

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import ContainerTracker

        tracker = ContainerTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import ContainerTracker

        tracker = ContainerTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None

    def test_process_message_returns_none_for_short_data(self) -> None:
        """Test process_message returns None for short data."""
        from tankpit_bot.sniffer import ContainerTracker

        tracker = ContainerTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        payload = base64.b64encode(b"\x02\x00\x2e").decode()
        result = tracker.process_message(payload)
        assert result is None


# =============================================================================
# TankExitTracker Tests
# =============================================================================


class TestTankExitTracker:
    """Tests for TankExitTracker class."""

    def test_init(self) -> None:
        """Test TankExitTracker initialization."""
        from tankpit_bot.sniffer import TankExitTracker

        tracker = TankExitTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._exited == set()

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import TankExitTracker

        tracker = TankExitTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_exited_tanks_property(self) -> None:
        """Test exited_tanks property returns copy of set."""
        from tankpit_bot.sniffer import TankExitTracker

        tracker = TankExitTracker()
        tracker._exited = {1, 2, 3}
        result = tracker.exited_tanks
        assert result == {1, 2, 3}
        # Verify it's a copy
        result.add(4)
        assert 4 not in tracker._exited

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import TankExitTracker

        tracker = TankExitTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import TankExitTracker

        tracker = TankExitTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


# =============================================================================
# EquipmentGainTracker Tests
# =============================================================================


class TestEquipmentGainTracker:
    """Tests for EquipmentGainTracker class."""

    def test_init(self) -> None:
        """Test EquipmentGainTracker initialization."""
        from tankpit_bot.sniffer import EquipmentGainTracker

        tracker = EquipmentGainTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import EquipmentGainTracker

        tracker = EquipmentGainTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import EquipmentGainTracker

        tracker = EquipmentGainTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00\x00\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import EquipmentGainTracker

        tracker = EquipmentGainTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


# =============================================================================
# FuelDepositTracker Tests
# =============================================================================


class TestFuelDepositTracker:
    """Tests for FuelDepositTracker class."""

    def test_init(self) -> None:
        """Test FuelDepositTracker initialization."""
        from tankpit_bot.sniffer import FuelDepositTracker

        tracker = FuelDepositTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._total_deposited == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import FuelDepositTracker

        tracker = FuelDepositTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_total_deposited_property(self) -> None:
        """Test total_deposited property returns count."""
        from tankpit_bot.sniffer import FuelDepositTracker

        tracker = FuelDepositTracker()
        assert tracker.total_deposited == 0
        tracker._total_deposited = 500
        assert tracker.total_deposited == 500

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import FuelDepositTracker

        tracker = FuelDepositTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import FuelDepositTracker

        tracker = FuelDepositTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


# =============================================================================
# RadarAckTracker Tests
# =============================================================================


class TestRadarAckTracker:
    """Tests for RadarAckTracker class."""

    def test_init(self) -> None:
        """Test RadarAckTracker initialization."""
        from tankpit_bot.sniffer import RadarAckTracker

        tracker = RadarAckTracker()
        assert tracker._xor_table is None
        assert tracker._static_key is None
        assert tracker._count == 0

    def test_set_magic_builds_xor_table(self) -> None:
        """Test set_magic builds XOR table from static key."""
        from tankpit_bot.sniffer import RadarAckTracker

        tracker = RadarAckTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")

        xor_table = tracker._xor_table
        if xor_table is None:
            raise AssertionError("_xor_table was not populated after set_magic")
        assert len(xor_table) == 1000

    def test_count_property(self) -> None:
        """Test count property returns acknowledgement count."""
        from tankpit_bot.sniffer import RadarAckTracker

        tracker = RadarAckTracker()
        assert tracker.count == 0
        tracker._count = 5
        assert tracker.count == 5

    def test_process_message_returns_none_without_magic(self) -> None:
        """Test process_message returns None when XOR table not set."""
        from tankpit_bot.sniffer import RadarAckTracker

        tracker = RadarAckTracker()
        payload = _make_payload(b"\x2e\x00\x00\x00")
        result = tracker.process_message(payload)
        assert result is None

    def test_process_message_returns_none_for_invalid_base64(self) -> None:
        """Test process_message returns None for invalid base64."""
        from tankpit_bot.sniffer import RadarAckTracker

        tracker = RadarAckTracker()
        tracker.set_magic("kp8ffxx7muk63a0ywtqh")
        result = tracker.process_message("not valid base64!!!")
        assert result is None


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestIsValidBase64:
    """Tests for _is_valid_base64 function."""

    def test_valid_base64(self) -> None:
        """Test valid base64 strings are recognized."""
        from tankpit_bot.sniffer import _is_valid_base64

        assert _is_valid_base64("SGVsbG8=") is True
        assert _is_valid_base64("YWJjZA==") is True
        assert _is_valid_base64("AAAA") is True

    def test_empty_string_invalid(self) -> None:
        """Test empty string is invalid."""
        from tankpit_bot.sniffer import _is_valid_base64

        assert _is_valid_base64("") is False

    def test_invalid_characters(self) -> None:
        """Test strings with invalid characters are rejected."""
        from tankpit_bot.sniffer import _is_valid_base64

        assert _is_valid_base64("invalid!@#$") is False
        assert _is_valid_base64("has space") is False

    def test_wrong_length(self) -> None:
        """Test strings with wrong length (not multiple of 4) are rejected."""
        from tankpit_bot.sniffer import _is_valid_base64

        assert _is_valid_base64("ABC") is False
        assert _is_valid_base64("ABCDE") is False


class TestDecodeBase64Safe:
    """Tests for _decode_base64_safe function."""

    def test_valid_base64(self) -> None:
        """Test valid base64 is decoded."""
        from tankpit_bot.sniffer import _decode_base64_safe

        result = _decode_base64_safe("SGVsbG8=")
        assert result == b"Hello"

    def test_invalid_base64_returns_none(self) -> None:
        """Test invalid base64 returns None."""
        from tankpit_bot.sniffer import _decode_base64_safe

        result = _decode_base64_safe("not valid!!!")
        assert result is None


class TestFormatSigKey:
    """Tests for _format_sig_key function."""

    def test_printable_ascii(self) -> None:
        """Test printable ASCII characters are shown."""
        from tankpit_bot.sniffer import _format_sig_key

        result = _format_sig_key(0x41)
        assert result == "0x41 'A'"

    def test_non_printable(self) -> None:
        """Test non-printable characters show question mark."""
        from tankpit_bot.sniffer import _format_sig_key

        result = _format_sig_key(0x01)
        assert result == "0x01 '?'"


class TestRankName:
    """Tests for _rank_name function."""

    def test_known_ranks(self) -> None:
        """Test known rank values return correct names."""
        from tankpit_bot.sniffer import _rank_name

        assert _rank_name(0) == "recruit"
        assert _rank_name(1) == "private"
        assert _rank_name(2) == "corporal"
        assert _rank_name(3) == "sergeant"
        assert _rank_name(4) == "lieutenant"
        assert _rank_name(5) == "captain"
        assert _rank_name(6) == "major"
        assert _rank_name(7) == "general"

    def test_unknown_rank(self) -> None:
        """Test unknown rank values return formatted string."""
        from tankpit_bot.sniffer import _rank_name

        assert _rank_name(8) == "r8"
        assert _rank_name(99) == "r99"

    def test_negative_rank(self) -> None:
        """Test negative rank values return formatted string."""
        from tankpit_bot.sniffer import _rank_name

        assert _rank_name(-1) == "r-1"


class TestDamageName:
    """Tests for _damage_name function."""

    def test_known_damage_states(self) -> None:
        """Test known damage values return correct names."""
        from tankpit_bot.sniffer import _damage_name

        assert _damage_name(0) == "full"
        assert _damage_name(1) == "light"
        assert _damage_name(2) == "medium"
        assert _damage_name(3) == "critical"

    def test_unknown_damage(self) -> None:
        """Test unknown damage values return formatted string."""
        from tankpit_bot.sniffer import _damage_name

        assert _damage_name(4) == "d4"
        assert _damage_name(99) == "d99"


class TestTeamName:
    """Tests for _team_name function."""

    def test_known_teams(self) -> None:
        """Test known team values return correct names."""
        from tankpit_bot.sniffer import _team_name

        assert _team_name(0) == "red"
        assert _team_name(1) == "blue"
        assert _team_name(2) == "green"
        assert _team_name(3) == "purple"

    def test_unknown_team(self) -> None:
        """Test unknown team values return formatted string."""
        from tankpit_bot.sniffer import _team_name

        assert _team_name(4) == "t4"
        assert _team_name(99) == "t99"


class TestEmptyMessageStats:
    """Tests for _empty_message_stats function."""

    def test_returns_empty_stats(self) -> None:
        """Test returns MessageStats with empty values."""
        from tankpit_bot.sniffer import _empty_message_stats

        result = _empty_message_stats()
        assert result["decoded"] == {}
        assert result["unknown"] == {}
        assert result["total_received"] == 0
        assert result["decode_coverage"] == "0%"


class TestXorDecode:
    """Tests for _xor_decode function."""

    def test_decode_with_no_global_table(self) -> None:
        """Test _xor_decode returns body[1:] when no global table."""
        from tankpit_bot import sniffer

        # Save original and reset
        original = sniffer._global_xor_table
        sniffer._global_xor_table = None

        result = sniffer._xor_decode(b"\x2e\x01\x02\x03")
        assert result == b"\x01\x02\x03"

        # Restore
        sniffer._global_xor_table = original

    def test_decode_short_body(self) -> None:
        """Test _xor_decode handles short body."""
        from tankpit_bot import sniffer

        # Save original and reset
        original = sniffer._global_xor_table
        sniffer._global_xor_table = None

        result = sniffer._xor_decode(b"\x2e")
        assert result == b""

        # Restore
        sniffer._global_xor_table = original


class TestDecode8ByteState:
    """Tests for _decode_8byte_state function."""

    def test_item_pickup_subtype(self) -> None:
        """Test 0x49 subtype returns ITEM_PICKUP."""
        from tankpit_bot.sniffer import _decode_8byte_state

        body = bytes([0x2E, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        result = _decode_8byte_state(body, "RECV")
        assert "[RECV] ITEM_PICKUP:" in result

    def test_game_state_subtype(self) -> None:
        """Test 0x67 subtype returns GAME_STATE."""
        from tankpit_bot.sniffer import _decode_8byte_state

        body = bytes([0x2E, 0x67, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        result = _decode_8byte_state(body, "RECV")
        assert "[RECV] GAME_STATE:" in result

    def test_unknown_subtype(self) -> None:
        """Test unknown subtype returns MSG_8B."""
        from tankpit_bot.sniffer import _decode_8byte_state

        body = bytes([0x2E, 0x99, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        result = _decode_8byte_state(body, "RECV")
        assert "[RECV] MSG_8B: sub=0x99" in result


class TestDecodeStateMessageFuelRaw:
    """Tests for _decode_state_message FUEL_RAW branch."""

    def test_fuel_raw_17_bytes_subtype_0x10(self) -> None:
        """Test 17-byte message with subtype 0x10 returns FUEL_RAW."""
        from tankpit_bot.sniffer import _decode_state_message

        # 17 bytes with subtype 0x10
        body = bytes([0x2E, 0x10]) + bytes(13) + bytes([0xE8, 0x03])
        result = _decode_state_message(body, "RECV")
        assert "[RECV] FUEL_RAW:" in result
        assert "p15=1000" in result


class TestExtractMessageSignature:
    """Tests for _extract_message_signature function."""

    def test_returns_none_for_invalid_base64(self) -> None:
        """Test returns None for invalid base64."""
        from tankpit_bot.sniffer import _extract_message_signature

        result = _extract_message_signature("not valid!!!", b"\x00" * 100)
        assert result is None

    def test_returns_none_when_no_dot_in_first_3_bytes(self) -> None:
        """Test returns None when no dot found in first 3 bytes."""
        from tankpit_bot.sniffer import _extract_message_signature

        payload = base64.b64encode(b"ABCDEFGH").decode()
        result = _extract_message_signature(payload, b"\x00" * 100)
        assert result is None

    def test_returns_none_when_dot_at_position_3_or_higher(self) -> None:
        """Test returns None when dot position >= 3."""
        from tankpit_bot.sniffer import _extract_message_signature

        payload = base64.b64encode(b"ABC.EFGH").decode()
        result = _extract_message_signature(payload, b"\x00" * 100)
        assert result is None

    def test_returns_none_when_no_data_after_dot(self) -> None:
        """Test returns None when no data after dot."""
        from tankpit_bot.sniffer import _extract_message_signature

        payload = base64.b64encode(b".").decode()
        result = _extract_message_signature(payload, b"\x00" * 100)
        assert result is None

    def test_decodes_with_xor_table(self) -> None:
        """Test decodes data using XOR table."""
        from tankpit_bot.sniffer import _extract_message_signature

        # Payload with dot at position 0
        raw_data = bytes([0x2E, 0x41, 0x42, 0x43])
        payload = base64.b64encode(raw_data).decode()
        xor_table = bytes([0x00, 0x00, 0x00])  # Identity XOR

        result = _extract_message_signature(payload, xor_table)
        assert result == bytes([0x41, 0x42, 0x43])


class TestMainPreferAccount:
    """Tests for main() with prefer_account option."""

    def test_main_prefer_account_enabled(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """Test main() enables prefer_account via env var."""
        _test_hooks.sync_playwright = fake_sync_playwright

        fake_env.set("TANKPIT_PREFER_ACCOUNT", "true")
        main()  # Should not raise

        fake_env.set("TANKPIT_PREFER_ACCOUNT", "1")
        main()  # Should not raise

        fake_env.set("TANKPIT_PREFER_ACCOUNT", "yes")
        main()  # Should not raise


# =============================================================================
# Tracker Process Message Tests (with XOR encoding)
# =============================================================================


def _build_test_xor_table(static_key: str, magic: str) -> bytes:
    """Build XOR table for testing.

    Args:
        static_key: Static key string.
        magic: Magic key string.

    Returns:
        XOR encoding table bytes.
    """
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
    return bytes(table)


def _make_tracker_payload(body: bytes) -> str:
    """Wrap a body in length header and base64 encode.

    The body should already be properly encoded (with XOR if needed).

    Args:
        body: Raw message body bytes.

    Returns:
        Base64 encoded payload with 2-byte length header.
    """
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


def _xor_encode_bytes(data: bytes, xor_table: bytes) -> bytes:
    """XOR encode bytes with table.

    Args:
        data: Data to encode.
        xor_table: XOR table.

    Returns:
        XOR encoded bytes.
    """
    result = bytearray(len(data))
    for i in range(len(data)):
        if i < len(xor_table):
            result[i] = data[i] ^ xor_table[i]
        else:
            result[i] = data[i]
    return bytes(result)


class TestDeactivationTrackerProcessMessage:
    """Tests for DeactivationTracker.process_message with XOR decoding."""

    def test_process_message_returns_kill_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes kill events correctly."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import DeactivationTracker

        # Set up static key
        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = DeactivationTracker()
        tracker.set_magic(magic)

        # Build XOR table
        xor_table = _build_test_xor_table(static_key, magic)

        # Deactivation format: body = [0x2E, XOR(0x41, victim_lo, victim_hi,
        #                                        killer_lo, killer_hi, extra, extra)]
        # Body must be 8 bytes. Victim ID = 100 (0x0064), Killer ID = 200 (0x00C8)
        decoded_data = bytes([0x41, 0x64, 0x00, 0xC8, 0x00, 0x00, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data  # 8 bytes total

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "KILL" in result
        assert "100" in result or "Tank" in result

    def test_process_message_returns_death_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message detects own death."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import DeactivationTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = DeactivationTracker()
        tracker.set_magic(magic)
        tracker.set_my_tank_id(100)  # Set our tank ID

        xor_table = _build_test_xor_table(static_key, magic)

        # Victim = 100 (our tank), Killer = 200
        decoded_data = bytes([0x41, 0x64, 0x00, 0xC8, 0x00, 0x00, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "DEATH" in result

    def test_process_message_returns_none_for_wrong_signature(
        self, fake_fs: FakeFileSystem
    ) -> None:
        """Test returns None when decoded signature is not 0x41."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import DeactivationTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = DeactivationTracker()
        tracker.set_magic(magic)

        xor_table = _build_test_xor_table(static_key, magic)

        # Use wrong signature 0x99 instead of 0x41
        decoded_data = bytes([0x99, 0x64, 0x00, 0xC8, 0x00, 0x00, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result is None

    def test_load_static_key_caches_result(self, fake_fs: FakeFileSystem) -> None:
        """Test _load_static_key caches the result."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import DeactivationTracker

        static_key = "CACHED_KEY_TEST" + "A" * 985
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        tracker = DeactivationTracker()

        # First call loads from file
        result1 = tracker._load_static_key()
        assert result1 == static_key

        # Second call should return cached value (even if we "change" the file)
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, "DIFFERENT_KEY" + "A" * 987)
        result2 = tracker._load_static_key()
        assert result2 == static_key  # Should be cached


class TestItemPickupTrackerProcessMessage:
    """Tests for ItemPickupTracker.process_message with XOR decoding."""

    def test_process_message_returns_pickup_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes item pickup correctly."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import ItemPickupTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ItemPickupTracker()
        tracker.set_magic(magic)

        xor_table = _build_test_xor_table(static_key, magic)

        # Pickup format: body = [0x2E, XOR(0x67, 0x01, armor, dual, missile, homing, radar)]
        # Body must be 8 bytes. Pick up 1 armor, 0 dual, 2 missiles, 1 homing, 0 radar
        decoded_data = bytes([0x67, 0x01, 0x01, 0x00, 0x02, 0x01, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data  # 8 bytes

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "PICKUP" in result
        assert "armor" in result
        assert "missile" in result

    def test_process_message_returns_none_for_all_zeros(self, fake_fs: FakeFileSystem) -> None:
        """Test returns None when all quantities are zero."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import ItemPickupTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ItemPickupTracker()
        tracker.set_magic(magic)

        xor_table = _build_test_xor_table(static_key, magic)

        # All zeros - still 8 byte body with 0x2E prefix
        decoded_data = bytes([0x67, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        encoded_data = _xor_encode_bytes(decoded_data, xor_table)
        body = bytes([0x2E]) + encoded_data

        payload = _make_tracker_payload(body)
        result = tracker.process_message(payload)
        assert result is None


class TestRadarTrackerProcessMessage:
    """Tests for RadarTracker.process_message with XOR decoding."""

    def test_classify_entity_fuel(self) -> None:
        """Test _classify_entity for fuel containers."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(50, 60, 100)
        assert category == "fuel"
        assert formatted == "(50,60)=100"

    def test_classify_entity_tank(self) -> None:
        """Test _classify_entity for tanks (0xFFFF)."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        category, formatted = tracker._classify_entity(70, 80, 0xFFFF)
        assert category == "tanks"
        assert formatted == "(70,80)"

    def test_classify_entity_equipment(self) -> None:
        """Test _classify_entity for equipment (negative values)."""
        from tankpit_bot.sniffer import RadarTracker

        tracker = RadarTracker()
        # Values >= 0x8000 are treated as negative (equipment)
        category, _formatted = tracker._classify_entity(30, 40, 0x8001)
        assert category == "equip"


class TestTankTrackerProcessMessage:
    """Tests for TankTracker._parse_* methods."""

    def test_parse_tank_join(self) -> None:
        """Test _parse_tank_join parses tank join message."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [subtype, tank_id_lo, tank_id_hi, extra...]
        decoded = bytearray([0x01, 0x64, 0x00, 0xAB, 0xCD])
        result = tracker._parse_tank_join(decoded)
        assert result, "Expected non-None result from _parse_tank_join"
        assert "JOIN" in result
        assert "100" in result or "id=100" in result

    def test_parse_tank_leave(self) -> None:
        """Test _parse_tank_leave parses tank leave message."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        decoded = bytearray([0x01, 0x64, 0x00, 0xAB, 0xCD])
        result = tracker._parse_tank_leave(decoded)
        assert result, "Expected non-None result from _parse_tank_leave"
        assert "LEAVE" in result

    def test_parse_tank_status(self) -> None:
        """Test _parse_tank_status parses and stores tank info."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [info_byte, tank_id_lo, tank_id_hi, 10 bytes, name...]
        # info_byte: team in bits 0-1, rank in bits 4-6
        # Team=1 (purple), Rank=3 (sergeant) -> 0x31
        decoded = bytearray([0x31, 0x64, 0x00]) + bytearray(10) + bytearray(b"TestPlayer")
        result = tracker._parse_tank_status(decoded)
        assert result, "Expected non-None result from _parse_tank_status"
        assert "STATUS" in result
        assert "sergeant" in result or "TestPlayer" in result

    def test_parse_movement(self) -> None:
        """Test _parse_movement parses tank movement."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [tank_id_lo, tank_id_hi, x, y, direction]
        decoded = bytearray([0x64, 0x00, 0x32, 0x3C, 0x02])
        result = tracker._parse_movement(decoded)
        assert result, "Expected non-None result from _parse_movement"
        assert "MOVE" in result
        assert "(50,60)" in result

    def test_parse_shooting(self) -> None:
        """Test _parse_shooting parses shot events."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [team, shooter_id_lo, shooter_id_hi, x, y]
        decoded = bytearray([0x01, 0x64, 0x00, 0x32, 0x3C])
        result = tracker._parse_shooting(decoded)
        assert result, "Expected non-None result from _parse_shooting"
        assert "SHOT" in result

    def test_parse_tank_info(self) -> None:
        """Test _parse_tank_info registers tank name."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [team, tank_id_lo, tank_id_hi, 7 bytes, name...]
        decoded = bytearray([0x01, 0x64, 0x00]) + bytearray(7) + bytearray(b"PlayerName")
        result = tracker._parse_tank_info(decoded)
        assert result, "Expected non-None result from _parse_tank_info"
        assert "INFO" in result
        assert "PlayerName" in result
        # Verify name was registered
        assert tracker.get_name(100) == "PlayerName"

    def test_parse_player_list(self) -> None:
        """Test _parse_player_list parses player list."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [tank_id_lo, tank_id_hi, b2, b3, b4]
        decoded = bytearray([0x64, 0x00, 0x01, 0x02, 0x03])
        result = tracker._parse_player_list(decoded)
        assert "PLAYERS" in result

    def test_parse_player_update(self) -> None:
        """Test _parse_player_update parses player updates."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: repeating [tank_id_lo, tank_id_hi, data]
        decoded = bytearray([0x64, 0x00, 0x01, 0xC8, 0x00, 0x02])
        result = tracker._parse_player_update(decoded)
        assert "PLAYERS" in result

    def test_parse_statistics(self) -> None:
        """Test _parse_statistics parses stats message."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [hours_lo, hours_hi, mins, secs, pad(3), destroyed, deactivated, pad(3), promo]
        decoded = bytearray(
            [
                0x05,
                0x00,  # 5 hours
                0x1E,
                0x0A,  # 30 mins, 10 secs
                0x00,
                0x00,
                0x00,  # padding
                0x10,
                0x08,  # destroyed=16, deactivated=8
                0x00,
                0x00,
                0x00,  # padding
                0x00,
                0x64,  # promo_pts=100
            ]
        )
        result = tracker._parse_statistics(decoded)
        assert "STATS" in result
        assert "5h" in result

    def test_parse_promotion(self) -> None:
        """Test _parse_promotion parses promotion message."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [rank, promoted_flag]
        decoded = bytearray([0x04, 0x01])  # Promoted to lieutenant
        result = tracker._parse_promotion(decoded)
        assert "PROMOTED" in result
        assert "lieutenant" in result

    def test_parse_promotion_demoted(self) -> None:
        """Test _parse_promotion handles demotion."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        decoded = bytearray([0x03, 0x00])  # Demoted to sergeant
        result = tracker._parse_promotion(decoded)
        assert "DEMOTED" in result

    def test_parse_supervisor_msg(self) -> None:
        """Test _parse_supervisor_msg parses supervisor message."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        # decoded: [0x01, 0x00, status]
        decoded = bytearray([0x01, 0x00, 0x04])
        result = tracker._parse_supervisor_msg(decoded)
        assert "SUPERVISOR" in result

    def test_get_all_names_returns_registered_names(self) -> None:
        """Test get_all_names returns all registered tank names."""
        from tankpit_bot.sniffer import TankTracker

        tracker = TankTracker()
        tracker.register_name(100, "Player1")
        tracker.register_name(200, "Player2")

        names = tracker.get_all_names()
        assert names == {100: "Player1", 200: "Player2"}


class TestMineTrackerParseMethods:
    """Tests for MineTracker._parse_* methods."""

    def test_parse_mine_placed(self) -> None:
        """Test _parse_mine_placed parses mine placement."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        # decoded: [0x4B, owner_id_lo, owner_id_hi, x, y, ...]
        decoded = bytearray([0x4B, 0x64, 0x00, 0x32, 0x3C, 0x00, 0x00])
        result = tracker._parse_mine_placed(decoded)
        assert "PLACED" in result
        assert tracker.mines_placed == 1

    def test_parse_mine_detonation(self) -> None:
        """Test _parse_mine_detonation parses mine explosions."""
        from tankpit_bot.sniffer import MineTracker

        tracker = MineTracker()
        # decoded: [0x45, count, x1, y1, x2, y2]
        decoded = bytearray([0x45, 0x02, 0x32, 0x3C, 0x33, 0x3D])
        result = tracker._parse_mine_detonation(decoded)
        assert "EXPLODE" in result
        assert "2 mines" in result
        assert tracker.mines_detonated == 2


def _build_xor_table(static_key: str, magic: str) -> bytes:
    """Build XOR table from static key and magic.

    Args:
        static_key: Static XOR key string.
        magic: Session magic key string.

    Returns:
        XOR table as bytes.
    """
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
    return bytes(table)


def _make_xor_payload(decoded_data: bytes, xor_table: bytes) -> str:
    """Create XOR-encoded base64 payload for testing.

    Args:
        decoded_data: The data after XOR decoding (what we want to test).
        xor_table: The XOR table to use for encoding.

    Returns:
        Base64 encoded payload with length header and 0x2E prefix.
    """
    # XOR encode the data
    encoded = bytes(decoded_data[i] ^ xor_table[i] for i in range(len(decoded_data)))
    # Add 0x2E prefix (not XOR encoded)
    body = bytes([0x2E]) + encoded
    # Add length header
    header = len(body).to_bytes(2, "little")
    return base64.b64encode(header + body).decode()


class TestEquipmentToggleTrackerParseMethods:
    """Tests for EquipmentToggleTracker methods."""

    def test_decode_toggle_with_xor_table(self, fake_fs: FakeFileSystem) -> None:
        """Test _decode_toggle works with proper setup."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import EquipmentToggleTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = EquipmentToggleTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # Toggle format decoded: 0x74 armor dual missile homing radar
        # armor=ON, dual=OFF, missile=ON, homing=OFF, radar=ON
        decoded_data = bytes([0x74, 0x01, 0x00, 0x01, 0x00, 0x01])
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "EQUIP" in result

    def test_state_property_returns_current_state(self, fake_fs: FakeFileSystem) -> None:
        """Test state property returns current equipment state."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import EquipmentToggleTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = EquipmentToggleTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        decoded_data = bytes([0x74, 0x01, 0x00, 0x01, 0x00, 0x01])
        payload = _make_xor_payload(decoded_data, xor_table)
        tracker.process_message(payload)

        state = tracker.state
        expected = {
            "armor": True,
            "dual": False,
            "missile": True,
            "homing": False,
            "radar": True,
        }
        assert state == expected

    def test_detect_changes_returns_changes(self, fake_fs: FakeFileSystem) -> None:
        """Test _detect_changes identifies equipment state changes.

        The tracker compares new state with _prev_state, which is set to
        the state from two messages ago. So changes are detected relative
        to that baseline, not the immediately previous message.
        """
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import EquipmentToggleTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = EquipmentToggleTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # First message: all OFF (sets baseline for next comparison)
        payload1 = _make_xor_payload(bytes([0x74, 0x00, 0x00, 0x00, 0x00, 0x00]), xor_table)
        tracker.process_message(payload1)

        # Second message: dual ON (compared to initial all-OFF state)
        payload2 = _make_xor_payload(bytes([0x74, 0x00, 0x01, 0x00, 0x00, 0x00]), xor_table)
        result = tracker.process_message(payload2)

        assert result, "Expected non-None result from process_message"
        assert "TOGGLE" in result
        assert "dual=ON" in result


class TestContainerTrackerProcessMessage:
    """Tests for ContainerTracker.process_message with XOR decoding."""

    def test_process_message_returns_container_info(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes container events correctly."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import ContainerTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ContainerTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # Container format decoded: 0x43 container_id_lo container_id_hi fuel_lo fuel_hi
        decoded_data = bytes([0x43, 0x64, 0x00, 0xE8, 0x03])  # id=100, fuel=1000
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "CONTAINER" in result
        assert "100" in result
        assert "1000" in result

    def test_container_depleted(self, fake_fs: FakeFileSystem) -> None:
        """Test container tracker handles depleted containers."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import ContainerTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ContainerTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # First: container has fuel
        payload1 = _make_xor_payload(bytes([0x43, 0x64, 0x00, 0xE8, 0x03]), xor_table)
        tracker.process_message(payload1)

        # Second: container depleted
        payload2 = _make_xor_payload(bytes([0x43, 0x64, 0x00, 0x00, 0x00]), xor_table)
        result = tracker.process_message(payload2)

        assert result, "Expected non-None result from process_message"
        assert "DEPLETED" in result

    def test_containers_property(self, fake_fs: FakeFileSystem) -> None:
        """Test containers property returns current state."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import ContainerTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = ContainerTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        payload = _make_xor_payload(bytes([0x43, 0x64, 0x00, 0xE8, 0x03]), xor_table)
        tracker.process_message(payload)

        containers = tracker.containers
        assert containers[100] == 1000


class TestTankExitTrackerProcessMessage:
    """Tests for TankExitTracker.process_message with XOR decoding."""

    def test_process_message_returns_exit_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes tank exit correctly."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import TankExitTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankExitTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # Tank exit format decoded: 0x58 tank_id_lo tank_id_hi
        decoded_data = bytes([0x58, 0x64, 0x00])  # tank_id=100
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "EXIT" in result
        assert "100" in result

    def test_exited_tanks_property(self, fake_fs: FakeFileSystem) -> None:
        """Test exited_tanks property tracks exited tanks."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import TankExitTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = TankExitTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        payload = _make_xor_payload(bytes([0x58, 0x64, 0x00]), xor_table)
        tracker.process_message(payload)

        assert 100 in tracker.exited_tanks


class TestEquipmentGainTrackerProcessMessage:
    """Tests for EquipmentGainTracker.process_message with XOR decoding."""

    def test_process_message_returns_gain_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes equipment gain correctly."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import EquipmentGainTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = EquipmentGainTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # Equipment gain decoded: 0x67 type zeros... equipment_flags
        # 7 bytes total for the decoded data
        decoded_data = bytes([0x67, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "EQUIP" in result or "GAIN" in result


class TestFuelDepositTrackerProcessMessage:
    """Tests for FuelDepositTracker.process_message with XOR decoding."""

    def test_process_message_returns_deposit_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes fuel deposit correctly."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import FuelDepositTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = FuelDepositTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # Fuel deposit decoded: 0x64 amount_lo amount_hi
        decoded_data = bytes([0x64, 0xE8, 0x03])  # amount=1000
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "FUEL" in result or "DEPOSIT" in result
        assert "1000" in result

    def test_total_deposited_property(self, fake_fs: FakeFileSystem) -> None:
        """Test total_deposited tracks cumulative deposits."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import FuelDepositTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = FuelDepositTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # Two deposits
        payload1 = _make_xor_payload(bytes([0x64, 0xE8, 0x03]), xor_table)
        payload2 = _make_xor_payload(bytes([0x64, 0xF4, 0x01]), xor_table)

        tracker.process_message(payload1)  # +1000
        tracker.process_message(payload2)  # +500

        assert tracker.total_deposited == 1500


class TestRadarAckTrackerProcessMessage:
    """Tests for RadarAckTracker.process_message with XOR decoding."""

    def test_process_message_returns_ack_event(self, fake_fs: FakeFileSystem) -> None:
        """Test process_message decodes radar ack correctly."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import RadarAckTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = RadarAckTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        # Radar ack decoded: 0x46 byte1 byte2
        decoded_data = bytes([0x46, 0x01, 0x00])
        payload = _make_xor_payload(decoded_data, xor_table)

        result = tracker.process_message(payload)
        assert result, "Expected non-None result from process_message"
        assert "RADAR" in result

    def test_radar_count_property(self, fake_fs: FakeFileSystem) -> None:
        """Test radar ack count is tracked."""
        from tankpit_bot.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import RadarAckTracker

        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)
        magic = "testmagic123"

        tracker = RadarAckTracker()
        tracker.set_magic(magic)

        xor_table = _build_xor_table(static_key, magic)

        payload = _make_xor_payload(bytes([0x46, 0x01, 0x00]), xor_table)
        tracker.process_message(payload)
        tracker.process_message(payload)

        assert tracker.count == 2


# =============================================================================
# Build Message Stats Tests
# =============================================================================


class TestBuildMessageStats:
    """Tests for _build_message_stats function."""

    def test_returns_empty_without_magic(self) -> None:
        """Test returns empty stats when session has no magic key."""
        from tankpit_bot.sniffer import _build_message_stats
        from tankpit_bot.types import CaptureSession

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic=None,
            game_log=[],
            tank_names={},
        )
        result = _build_message_stats(session)
        assert result["decoded"] == {}
        assert result["unknown"] == {}
        assert result["total_received"] == 0


class TestBuildSessionSummary:
    """Tests for _build_session_summary function."""

    def test_extracts_combat_events_from_game_log(self) -> None:
        """Test extracts combat events from game log."""
        from tankpit_bot.sniffer import _build_session_summary
        from tankpit_bot.types import CaptureSession, GameLogEntryWithTimestamp

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic="test",
            game_log=[
                GameLogEntryWithTimestamp(
                    timestamp_ms=100,
                    text="You hit Enemy",
                    category="combat",
                ),
                GameLogEntryWithTimestamp(
                    timestamp_ms=200,
                    text="You killed Enemy",
                    category="combat",
                ),
                GameLogEntryWithTimestamp(
                    timestamp_ms=300,
                    text="Foe hit you",
                    category="combat",
                ),
                GameLogEntryWithTimestamp(
                    timestamp_ms=400,
                    text="Foe killed you",
                    category="combat",
                ),
            ],
            tank_names={},
        )
        result = _build_session_summary(session)
        assert len(result["combat"]) == 4
        assert result["combat"][0]["event_type"] == "hit"
        assert result["combat"][0]["target"] == "Enemy"
        assert result["combat"][1]["event_type"] == "kill"
        assert result["combat"][2]["event_type"] == "hit_by"
        assert result["combat"][3]["event_type"] == "killed_by"

    def test_skips_non_combat_log_entries(self) -> None:
        """Test skips non-combat log entries."""
        from tankpit_bot.sniffer import _build_session_summary
        from tankpit_bot.types import CaptureSession, GameLogEntryWithTimestamp

        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic="test",
            game_log=[
                GameLogEntryWithTimestamp(
                    timestamp_ms=100,
                    text="Some info message",
                    category="info",
                ),
            ],
            tank_names={},
        )
        result = _build_session_summary(session)
        assert len(result["combat"]) == 0
