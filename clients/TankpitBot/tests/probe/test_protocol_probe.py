"""Tests for ProtocolProbe class."""

from __future__ import annotations

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.probe import (
    PlaywrightNotInstalledError,
    ProbeError,
    ProtocolProbe,
)
from tankpit_bot.types import CapturedMessage
from tests.conftest import FakeFileSystem
from tests.fakes import (
    fake_sync_playwright_probe,
    fake_sync_playwright_probe_no_messages,
)


def test_protocol_probe_init() -> None:
    """Test ProtocolProbe initialization."""
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    assert probe._target_url == "https://tankpit.com/play"
    assert probe._headless is True


def test_protocol_probe_run_without_playwright() -> None:
    """Test ProtocolProbe.run raises error when Playwright not installed."""
    _test_hooks.sync_playwright = None
    probe = ProtocolProbe("https://tankpit.com/play")
    with pytest.raises(PlaywrightNotInstalledError, match="Playwright is not installed"):
        probe.run(
            probe_keys=["s"],  # Use key with known command mapping
            probe_mouse_positions=[],
            wait_after_join_ms=1000,
            wait_after_input_ms=100,
        )


def test_protocol_probe_run_game_not_joined(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run raises error when no messages captured."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe_no_messages

    probe = ProtocolProbe("https://tankpit.com/play")
    with pytest.raises(ProbeError, match="Cannot build XOR table: magic key not captured"):
        probe.run(
            probe_keys=["s"],  # Use key with known command mapping
            probe_mouse_positions=[],
            wait_after_join_ms=1000,
            wait_after_input_ms=100,
        )


def test_protocol_probe_run_captures_key_inputs(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run captures messages from key inputs."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["s", "d"],  # Use keys with known command mappings
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert session["base_url"] == "https://tankpit.com/play"
    assert len(session["results"]) == 2

    # First result should be for key 's' (radar)
    first_result = session["results"][0]
    assert first_result["input"]["input_type"] == "key"
    key_input = first_result["input"]["key_input"]
    assert type(key_input) is dict
    assert key_input["key"] == "s"
    # Should have captured a sent message
    assert len(first_result["messages_after"]) == 1


def test_protocol_probe_run_captures_mouse_inputs(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run captures messages from mouse inputs."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    probe = ProtocolProbe("https://tankpit.com/play")
    session = probe.run(
        probe_keys=[],
        probe_mouse_positions=[(0.5, 0.5)],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert len(session["results"]) == 1

    # Result should be for mouse input
    result = session["results"][0]
    assert result["input"]["input_type"] == "mouse"
    mouse_input = result["input"]["mouse_input"]
    assert type(mouse_input) is dict
    # Viewport is 800x600, so 50% = 400, 300
    assert mouse_input["x"] == 400
    assert mouse_input["y"] == 300


def test_protocol_probe_run_mouse_emits_messages(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run captures messages from mouse inputs that emit."""
    from tests.fakes import fake_sync_playwright_probe_mouse_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_mouse_emits

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=[],
        probe_mouse_positions=[(0.5, 0.5)],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert len(session["results"]) == 1
    result = session["results"][0]
    assert result["input"]["input_type"] == "mouse"
    # Should have captured a sent message from mouse click
    assert len(result["messages_after"]) == 1


def test_protocol_probe_run_no_key_emit(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run when keys don't generate messages."""
    from tests.fakes import fake_sync_playwright_probe_no_key_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_no_key_emits

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["s"],  # Use key with known command mapping
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert len(session["results"]) == 1
    result = session["results"][0]
    assert result["input"]["input_type"] == "key"
    # No messages should be captured since emit_on_key is False
    assert len(result["messages_after"]) == 0


def test_protocol_probe_run_invalid_viewport(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run handles invalid viewport result gracefully."""
    from tests.fakes import fake_sync_playwright_probe_invalid_viewport

    _test_hooks.sync_playwright = fake_sync_playwright_probe_invalid_viewport

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["s"],  # Use key with known command mapping
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # Should complete successfully with default viewport
    assert len(session["results"]) == 1


def test_protocol_probe_run_non_dict_viewport(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run handles non-dict viewport result gracefully."""
    from tests.fakes import fake_sync_playwright_probe_non_dict_viewport

    _test_hooks.sync_playwright = fake_sync_playwright_probe_non_dict_viewport

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["s"],  # Use key with known command mapping
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # Should complete successfully with default viewport
    assert len(session["results"]) == 1


def test_protocol_probe_run_stabilization_reset(fake_fs: FakeFileSystem) -> None:
    """Test probe stabilization loop resets when new messages arrive.

    This covers the branch where stable_checks is reset to 0 when
    messages continue to arrive during the stabilization wait.
    """
    from tests.fakes import fake_sync_playwright_probe_delayed_messages

    _test_hooks.sync_playwright = fake_sync_playwright_probe_delayed_messages

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["s"],  # Use key with known command mapping
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # Should still complete successfully with extra message from stabilization
    assert len(session["results"]) == 1
    # Should have captured the extra message during stabilization
    assert len(probe._messages) > 2  # More than just initial auth + room_list


def test_protocol_probe_on_message_captured_non_auth_sent() -> None:
    """Test _on_message_captured handles sent message without AUTH magic."""
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)

    # Send a message that's not a valid AUTH payload (just random text)
    message: CapturedMessage = {
        "ws_url": "wss://tankpit.com/ws/",
        "timestamp_ms": 1000,
        "direction": "sent",
        "payload": "not_valid_base64!!",
    }

    # Should not raise, magic stays None
    probe._on_message_captured(message)
    assert probe._magic is None


def test_protocol_probe_build_xor_table_raises_without_magic(fake_fs: FakeFileSystem) -> None:
    """Test _build_xor_table raises ProbeError when magic is not captured."""
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    # Magic is None by default
    with pytest.raises(ProbeError, match="Cannot build XOR table: magic key not captured"):
        probe._build_xor_table()


def test_protocol_probe_encode_xor_command_raises_without_table(fake_fs: FakeFileSystem) -> None:
    """Test _encode_xor_command raises ProbeError when XOR table is not built."""
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    # XOR table is None by default
    with pytest.raises(ProbeError, match="XOR table not initialized"):
        probe._encode_xor_command(102)  # CMD_RADAR


def test_protocol_probe_send_key_command_unknown_key(
    fake_fs: FakeFileSystem,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test _send_key_command returns False for unknown keys."""
    import logging

    from tests.fakes import FakeCDPSessionProbe

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    cdp = FakeCDPSessionProbe()

    # 'w' is not in KEY_TO_COMMAND or KEY_TO_PLAIN_COMMAND
    with caplog.at_level(logging.WARNING):
        result = probe._send_key_command(cdp, "w")

    assert result == "UNKNOWN_KEY"
    assert "Unknown key: w" in caplog.text
