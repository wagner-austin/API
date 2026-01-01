"""Verify decoder by inspecting captured session data.

Run with: poetry run python scripts/verify_decode.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from platform_core.logging import get_logger

from tankpit_bot.decoder import DecodedCommand, DecodedLobbyMessage

from . import _test_hooks

log = get_logger(__name__)


def log_command_summary(commands: list[DecodedCommand]) -> None:
    """Log summary of commands by type and cmd byte.

    Args:
        commands: List of decoded commands.
    """
    log.info("=" * 60)
    log.info("COMMANDS BY TYPE")
    log.info("=" * 60)

    type_bytes: dict[int, list[int]] = {}
    for cmd in commands:
        tb = cmd["type_byte"]
        cb = cmd["cmd_byte"]
        if tb not in type_bytes:
            type_bytes[tb] = []
        if cb not in type_bytes[tb]:
            type_bytes[tb].append(cb)

    for tb in sorted(type_bytes.keys()):
        cmds_with_type = [c for c in commands if c["type_byte"] == tb]
        log.info("type_byte=0x%02x (%d commands)", tb, len(cmds_with_type))

        for cb in sorted(type_bytes[tb]):
            cmds_with_cb = [c for c in cmds_with_type if c["cmd_byte"] == cb]
            example = cmds_with_cb[0]
            data_len = len(example["data_hex"]) // 2
            log.info("  cmd=0x%02x: %dx, data_len=%d", cb, len(cmds_with_cb), data_len)


def log_command_details(commands: list[DecodedCommand]) -> None:
    """Log first 10 commands in detail.

    Args:
        commands: List of decoded commands.
    """
    log.info("=" * 60)
    log.info("FIRST 10 COMMANDS (DETAIL)")
    log.info("=" * 60)
    for i, cmd in enumerate(commands[:10]):
        log.info(
            "[%d] %s ts=%d type=0x%02x cmd=0x%02x",
            i,
            cmd["direction"],
            cmd["timestamp_ms"],
            cmd["type_byte"],
            cmd["cmd_byte"],
        )
        log.info("    raw=%s decoded=%s", cmd["raw_hex"], cmd["decoded_hex"])
        if cmd["data_hex"]:
            log.info("    data=%s", cmd["data_hex"])


def log_lobby_messages(messages: list[DecodedLobbyMessage]) -> None:
    """Log lobby message summary and details.

    Args:
        messages: List of decoded lobby messages.
    """
    log.info("=" * 60)
    log.info("LOBBY MESSAGES")
    log.info("=" * 60)

    # Group by prefix
    prefixes: dict[str, int] = {}
    for msg in messages:
        p = msg["prefix"]
        prefixes[p] = prefixes.get(p, 0) + 1

    log.info("By prefix:")
    for p, count in sorted(prefixes.items()):
        log.info("  '%s': %d messages", p, count)

    log.info("All lobby messages:")
    for msg in messages:
        direction = msg["direction"]
        prefix = msg["prefix"]
        text = msg["text"][:70]
        if len(msg["text"]) > 70:
            text += "..."
        log.info("  %s [%s] %s", direction, prefix, text)


def main() -> None:
    """Decode captured session and display results."""
    load_dotenv()
    _test_hooks.setup_rich_logging("INFO")

    session_path = Path("capture_session.json")

    if not _test_hooks.path_exists(session_path):
        log.error("File not found: %s", session_path)
        return

    log.info("Loading: %s", session_path)
    decoder = _test_hooks.load_and_decode_session(session_path)

    log.info("Decoded %d commands", len(decoder.commands))
    log.info("Decoded %d lobby messages", len(decoder.lobby_messages))

    if decoder.commands:
        log_command_summary(decoder.commands)
        log_command_details(decoder.commands)

    if decoder.lobby_messages:
        log_lobby_messages(decoder.lobby_messages)


if __name__ == "__main__":
    main()
