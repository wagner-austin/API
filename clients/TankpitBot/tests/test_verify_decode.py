"""Tests for scripts.verify_decode module."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from scripts import _test_hooks
from scripts._test_hooks import (
    LoadAndDecodeSessionFunc,
    LogLevel,
    SessionDecoderProtocol,
)
from scripts.verify_decode import (
    log_command_details,
    log_command_summary,
    log_lobby_messages,
    main,
)

from tankpit_bot.decoder import DecodedCommand, DecodedLobbyMessage

# =============================================================================
# Test Data Factories
# =============================================================================


def make_sample_commands() -> list[DecodedCommand]:
    """Create sample decoded commands for testing.

    Returns:
        List of sample DecodedCommand objects.
    """
    return [
        DecodedCommand(
            timestamp_ms=1000,
            direction="sent",
            raw_hex="212f76",
            decoded_hex="0c660a",
            type_byte=0x66,
            cmd_byte=0x0A,
            data_hex="",
        ),
        DecodedCommand(
            timestamp_ms=2000,
            direction="sent",
            raw_hex="212939deb3",
            decoded_hex="0c6045c982",
            type_byte=0x60,
            cmd_byte=0x45,
            data_hex="c982",
        ),
        DecodedCommand(
            timestamp_ms=3000,
            direction="sent",
            raw_hex="212939dcb3",
            decoded_hex="0c6045cb82",
            type_byte=0x60,
            cmd_byte=0x45,
            data_hex="cb82",
        ),
    ]


def make_sample_lobby_messages() -> list[DecodedLobbyMessage]:
    """Create sample lobby messages for testing.

    Returns:
        List of sample DecodedLobbyMessage objects.
    """
    return [
        DecodedLobbyMessage(
            timestamp_ms=100,
            direction="sent",
            prefix="%",
            text="AUTH data here",
        ),
        DecodedLobbyMessage(
            timestamp_ms=200,
            direction="received",
            prefix="+",
            text="4|World|42|1,1,1|2|n|field.gif|2025",
        ),
        DecodedLobbyMessage(
            timestamp_ms=300,
            direction="received",
            prefix="+",
            text="A very long message that exceeds seventy characters - extra text truncated",
        ),
    ]


class FakeSessionDecoder:
    """Fake SessionDecoder for testing.

    Implements SessionDecoderProtocol interface.
    """

    def __init__(
        self,
        commands: list[DecodedCommand],
        lobby_messages: list[DecodedLobbyMessage],
    ) -> None:
        """Initialize with preset data.

        Args:
            commands: List of decoded commands.
            lobby_messages: List of decoded lobby messages.
        """
        self._commands = commands
        self._lobby_messages = lobby_messages

    @property
    def commands(self) -> list[DecodedCommand]:
        """Get commands."""
        return self._commands

    @property
    def lobby_messages(self) -> list[DecodedLobbyMessage]:
        """Get lobby messages."""
        return self._lobby_messages


# =============================================================================
# log_command_summary Tests
# =============================================================================


def test_log_command_summary_groups_by_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that commands are grouped by type_byte."""
    caplog.set_level(logging.INFO)
    log_command_summary(make_sample_commands())

    assert "type_byte=0x60" in caplog.text
    assert "type_byte=0x66" in caplog.text


def test_log_command_summary_groups_by_cmd(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that commands are grouped by cmd_byte within type."""
    caplog.set_level(logging.INFO)
    log_command_summary(make_sample_commands())

    assert "cmd=0x45" in caplog.text
    assert "cmd=0x0a" in caplog.text


def test_log_command_summary_shows_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that command counts are shown."""
    caplog.set_level(logging.INFO)
    log_command_summary(make_sample_commands())

    # 0x60/0x45 appears twice
    assert "2x" in caplog.text


# =============================================================================
# log_command_details Tests
# =============================================================================


def test_log_command_details_shows_first_10(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that first 10 commands are shown."""
    caplog.set_level(logging.INFO)
    log_command_details(make_sample_commands())

    assert "[0]" in caplog.text
    assert "[1]" in caplog.text
    assert "[2]" in caplog.text


def test_log_command_details_shows_hex(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that raw and decoded hex are shown."""
    caplog.set_level(logging.INFO)
    log_command_details(make_sample_commands())

    assert "raw=212f76" in caplog.text
    assert "decoded=0c660a" in caplog.text


def test_log_command_details_shows_data_when_present(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that data is shown when non-empty."""
    caplog.set_level(logging.INFO)
    log_command_details(make_sample_commands())

    assert "data=c982" in caplog.text


def test_log_command_details_skips_data_when_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that data line is skipped when empty."""
    caplog.set_level(logging.INFO)
    commands: list[DecodedCommand] = [
        DecodedCommand(
            timestamp_ms=1000,
            direction="sent",
            raw_hex="aabbcc",
            decoded_hex="112233",
            type_byte=0x11,
            cmd_byte=0x22,
            data_hex="",
        ),
    ]
    log_command_details(commands)

    # Should not have "data=" line for this command
    lines_with_data = [line for line in caplog.text.split("\n") if "data=" in line]
    assert len(lines_with_data) == 0


# =============================================================================
# log_lobby_messages Tests
# =============================================================================


def test_log_lobby_messages_groups_by_prefix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that messages are grouped by prefix."""
    caplog.set_level(logging.INFO)
    log_lobby_messages(make_sample_lobby_messages())

    assert "'%': 1 messages" in caplog.text
    assert "'+': 2 messages" in caplog.text


def test_log_lobby_messages_truncates_long_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that long messages are truncated."""
    caplog.set_level(logging.INFO)
    log_lobby_messages(make_sample_lobby_messages())

    assert "..." in caplog.text


def test_log_lobby_messages_shows_direction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that direction is shown."""
    caplog.set_level(logging.INFO)
    log_lobby_messages(make_sample_lobby_messages())

    assert "sent" in caplog.text
    assert "received" in caplog.text


# =============================================================================
# main Tests
# =============================================================================


def test_main_file_not_found(caplog: pytest.LogCaptureFixture) -> None:
    """Test main logs error when file not found."""
    caplog.set_level(logging.INFO)
    original_path_exists = _test_hooks.path_exists
    original_setup_logging = _test_hooks.setup_rich_logging

    def fake_path_exists(path: Path) -> bool:
        return False

    def fake_setup_logging(level: LogLevel) -> None:
        pass

    _test_hooks.path_exists = fake_path_exists
    _test_hooks.setup_rich_logging = fake_setup_logging

    main()

    _test_hooks.path_exists = original_path_exists
    _test_hooks.setup_rich_logging = original_setup_logging

    assert "File not found" in caplog.text


def test_main_with_commands_and_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main with commands and lobby messages."""
    caplog.set_level(logging.INFO)
    original_path_exists = _test_hooks.path_exists
    original_load = _test_hooks.load_and_decode_session
    original_setup_logging = _test_hooks.setup_rich_logging

    def fake_path_exists(path: Path) -> bool:
        return True

    def fake_setup_logging(level: LogLevel) -> None:
        pass

    fake_decoder: SessionDecoderProtocol = FakeSessionDecoder(
        make_sample_commands(), make_sample_lobby_messages()
    )

    def fake_load(path: Path) -> SessionDecoderProtocol:
        return fake_decoder

    _test_hooks.path_exists = fake_path_exists
    _test_hooks.setup_rich_logging = fake_setup_logging
    fake_load_typed: LoadAndDecodeSessionFunc = fake_load
    _test_hooks.load_and_decode_session = fake_load_typed

    main()

    _test_hooks.path_exists = original_path_exists
    _test_hooks.load_and_decode_session = original_load
    _test_hooks.setup_rich_logging = original_setup_logging

    assert "Decoded 3 commands" in caplog.text
    assert "Decoded 3 lobby messages" in caplog.text
    assert "COMMANDS BY TYPE" in caplog.text
    assert "LOBBY MESSAGES" in caplog.text


def test_main_no_commands(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main with no commands."""
    caplog.set_level(logging.INFO)
    original_path_exists = _test_hooks.path_exists
    original_load = _test_hooks.load_and_decode_session
    original_setup_logging = _test_hooks.setup_rich_logging

    def fake_path_exists(path: Path) -> bool:
        return True

    def fake_setup_logging(level: LogLevel) -> None:
        pass

    empty_commands: list[DecodedCommand] = []
    fake_decoder: SessionDecoderProtocol = FakeSessionDecoder(
        empty_commands, make_sample_lobby_messages()
    )

    def fake_load(path: Path) -> SessionDecoderProtocol:
        return fake_decoder

    _test_hooks.path_exists = fake_path_exists
    _test_hooks.setup_rich_logging = fake_setup_logging
    fake_load_typed: LoadAndDecodeSessionFunc = fake_load
    _test_hooks.load_and_decode_session = fake_load_typed

    main()

    _test_hooks.path_exists = original_path_exists
    _test_hooks.load_and_decode_session = original_load
    _test_hooks.setup_rich_logging = original_setup_logging

    assert "Decoded 0 commands" in caplog.text
    assert "COMMANDS BY TYPE" not in caplog.text
    assert "LOBBY MESSAGES" in caplog.text


def test_main_no_lobby_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test main with no lobby messages."""
    caplog.set_level(logging.INFO)
    original_path_exists = _test_hooks.path_exists
    original_load = _test_hooks.load_and_decode_session
    original_setup_logging = _test_hooks.setup_rich_logging

    def fake_path_exists(path: Path) -> bool:
        return True

    def fake_setup_logging(level: LogLevel) -> None:
        pass

    empty_messages: list[DecodedLobbyMessage] = []
    fake_decoder: SessionDecoderProtocol = FakeSessionDecoder(
        make_sample_commands(), empty_messages
    )

    def fake_load(path: Path) -> SessionDecoderProtocol:
        return fake_decoder

    _test_hooks.path_exists = fake_path_exists
    _test_hooks.setup_rich_logging = fake_setup_logging
    fake_load_typed: LoadAndDecodeSessionFunc = fake_load
    _test_hooks.load_and_decode_session = fake_load_typed

    main()

    _test_hooks.path_exists = original_path_exists
    _test_hooks.load_and_decode_session = original_load
    _test_hooks.setup_rich_logging = original_setup_logging

    assert "Decoded 0 lobby messages" in caplog.text
    assert "LOBBY MESSAGES" not in caplog.text
    assert "COMMANDS BY TYPE" in caplog.text


# =============================================================================
# _test_hooks Tests
# =============================================================================


def test_real_path_exists() -> None:
    """Test _real_path_exists returns correct result."""
    from scripts._test_hooks import _real_path_exists

    # Current directory should exist
    assert _real_path_exists(Path(".")) is True
    # Non-existent path should not exist
    assert _real_path_exists(Path("/nonexistent/path/xyz")) is False


def test_real_load_and_decode_session_file_not_found() -> None:
    """Test _real_load_and_decode_session raises on missing file."""
    from scripts._test_hooks import _real_load_and_decode_session

    with pytest.raises(FileNotFoundError):
        _real_load_and_decode_session(Path("/nonexistent/session.json"))


def test_real_setup_rich_logging() -> None:
    """Test _real_setup_rich_logging runs without error."""
    from scripts._test_hooks import _real_setup_rich_logging

    # Just verify it doesn't raise
    _real_setup_rich_logging("DEBUG")


def test_verify_decode_main_entrypoint() -> None:
    """Test that verify_decode runs as __main__."""
    import runpy
    import sys

    original_path_exists = _test_hooks.path_exists
    original_setup_logging = _test_hooks.setup_rich_logging

    def fake_path_exists(path: Path) -> bool:
        return False

    def fake_setup_logging(level: LogLevel) -> None:
        pass

    _test_hooks.path_exists = fake_path_exists
    _test_hooks.setup_rich_logging = fake_setup_logging

    # Clear module from sys.modules to avoid runpy warning
    mod_name = "scripts.verify_decode"
    saved_module = sys.modules.pop(mod_name, None)

    # Run the module as __main__
    runpy.run_module(mod_name, run_name="__main__")

    # Restore module if it was cached
    if saved_module is not None:
        sys.modules[mod_name] = saved_module

    _test_hooks.path_exists = original_path_exists
    _test_hooks.setup_rich_logging = original_setup_logging
