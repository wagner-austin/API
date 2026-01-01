"""Tests for tankpit_bot.sniffer module."""

from __future__ import annotations

import base64

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.sniffer import (
    SnifferError,
    WebSocketSniffer,
    _decode_command,
    _decode_join_confirm,
    _decode_message,
    _decode_plus_message,
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
    payload = _make_payload(b".binary state data here")
    result = _decode_message(payload, "received")
    assert "[RECEIVED] STATE: len=" in result


def test_decode_message_unknown() -> None:
    """Test _decode_message handles unknown message types."""
    payload = _make_payload(b"some unknown message format")
    result = _decode_message(payload, "received")
    assert "[RECEIVED] ???:" in result


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
    """Test _decode_command decodes commands with additional data."""
    # Use actual binary bytes, not ASCII string
    body = bytes([0x21, 0x31, 0x2D, 0x43, 0xFE])  # !1-C<0xFE>
    result = _decode_command(body, "!1...", "SENT")
    assert result == "[SENT] CMD: !1 2d43fe"


def test_decode_command_short() -> None:
    """Test _decode_command handles short command messages."""
    result = _decode_command(b"!", "!", "SENT")
    assert result == "[SENT] CMD: !"


def test_decode_command_non_ascii() -> None:
    """Test _decode_command handles non-ASCII command bytes."""
    body = bytes([0x21, 0x90, 0xAB, 0xCD])  # !<0x90><0xAB><0xCD>
    result = _decode_command(body, "!", "SENT")
    assert result == "[SENT] CMD: !0x90 abcd"


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
    assert result == "[SENT] CMD: !7 62"


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
    assert "Script: https://tankpit.com/js/game.js" in output
    assert "Script: https://tankpit.com/js/protocol.js" in output


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
    assert "Script: https://tankpit.com/js/valid.js" in output
    assert "Script: https://tankpit.com/js/another.js" in output
    # Should NOT log the non-string values (123, None)
    assert "Script: 123" not in output
    assert "Script: None" not in output


# =============================================================================
# Error Class Tests
# =============================================================================


def test_sniffer_error_is_exception() -> None:
    """Test SnifferError is an Exception."""
    assert issubclass(SnifferError, Exception)
    err = SnifferError("test error")
    assert str(err) == "test error"
