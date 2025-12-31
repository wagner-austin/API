"""Tests for tankpit_bot.bot module."""

from __future__ import annotations

import pytest

from tankpit_bot.bot import BotError, ProtocolNotDiscoveredError, main
from tests.conftest import FakeEnv


def test_main_prints_instructions(
    fake_env: FakeEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints usage instructions."""
    main()

    captured = capsys.readouterr()
    lines = captured.out.split("\n")
    assert lines[0] == "TankpitBot - Automated Tankpit.com player"
    assert lines[4] == "  1. Run the sniffer to capture WebSocket traffic:"
    assert lines[5] == "     make sniff"


def test_main_uses_custom_capture_path(
    fake_env: FakeEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() uses custom capture path from env."""
    fake_env.set("TANKPIT_CAPTURE", "custom_capture.json")

    main()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    # Find the line after "save the captured protocol to:"
    for i, line in enumerate(output_lines):
        if "save the captured protocol to:" in line:
            assert output_lines[i + 1].strip() == "custom_capture.json"
            return
    raise AssertionError("Expected 'save the captured protocol' line not found")


def test_main_default_capture_path(
    fake_env: FakeEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() uses default capture path when env not set."""
    main()

    captured = capsys.readouterr()
    output_lines = captured.out.strip().split("\n")
    # Find the line after "save the captured protocol to:"
    for i, line in enumerate(output_lines):
        if "save the captured protocol to:" in line:
            assert output_lines[i + 1].strip() == "capture_session.json"
            return
    raise AssertionError("Expected 'save the captured protocol' line not found")


def test_bot_error_is_exception() -> None:
    """Test BotError is an Exception."""
    assert issubclass(BotError, Exception)
    err = BotError("test error")
    assert str(err) == "test error"


def test_protocol_not_discovered_error_is_bot_error() -> None:
    """Test ProtocolNotDiscoveredError is a BotError."""
    assert issubclass(ProtocolNotDiscoveredError, BotError)
