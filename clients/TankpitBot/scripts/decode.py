"""Decode a captured WebSocket session and print all messages.

Usage: poetry run python -m scripts.decode [session.json]

Reads capture_session.json (or the given path), builds the XOR table from
the magic key, and decodes every message using the sniffer's decoder pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.sniffer.constants import TEXT_MESSAGE_TYPES
from tankpit_bot.sniffer.decoders import decode_text_message, try_decode_binary
from tankpit_bot.sniffer.xor import build_global_xor_table, xor_decode
from tankpit_bot.types import decode_capture_session

log = get_logger(__name__)


def _decode_sent(payload: str) -> str | None:
    """Decode a sent message payload.

    Args:
        payload: Base64-encoded message payload.

    Returns:
        Decoded message string, or None if invalid.
    """
    data = decode_base64_safe(payload)
    if data is None or len(data) < 3:
        return None

    body = data[2:]
    first = body[0]

    # Text messages
    if first in TEXT_MESSAGE_TYPES:
        text = body.decode("utf-8", errors="replace")
        return decode_text_message(text, len(body), "SENT", body)

    # XOR command (0x21 = '!')
    if first == 0x21:
        decoded = xor_decode(body)
        if len(decoded) < 2:
            return f"[SENT] CMD: len={len(body)} hex={body.hex()}"
        msg_type = decoded[0]
        return "[SENT] " + try_decode_binary(msg_type, decoded, body)

    return f"[SENT] RAW: len={len(body)} hex={body[:20].hex()}"


def _decode_received(payload: str) -> str | None:
    """Decode a received message payload.

    Args:
        payload: Base64-encoded message payload.

    Returns:
        Decoded message string, or None if invalid.
    """
    data = decode_base64_safe(payload)
    if data is None or len(data) < 3:
        return None

    body = data[2:]
    msg_type = body[0]

    # Text messages
    if msg_type in TEXT_MESSAGE_TYPES:
        text = body.decode("utf-8", errors="replace")
        return decode_text_message(text, len(body), "RECEIVED", body)

    # Binary messages - XOR decode
    decoded_data = xor_decode(body)
    if len(decoded_data) == 0:
        return f"[RECEIVED] EMPTY: type=0x{msg_type:02X}"

    return "[RECEIVED] " + try_decode_binary(msg_type, decoded_data, body)


def main() -> None:
    """Decode and print all messages from a capture session."""
    setup_rich_logging(level="INFO")

    session_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("capture_session.json")
    if not _test_hooks.path_exists(session_path):
        log.error("File not found: %s", session_path)
        sys.exit(1)

    # Load session
    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    session_text = _test_hooks.read_text(session_path)
    session_json = narrow_json_to_dict(load_json_str(session_text))
    session = decode_capture_session(session_json)

    magic = session["magic"]
    messages = session["messages"]
    log.info("Session: %s", session["session_id"])
    log.info("Messages: %d", len(messages))
    log.info("Magic: %s", magic or "(none)")

    if magic is None:
        log.error("No magic key in session — cannot XOR-decode binary messages")
        sys.exit(1)

    # Build XOR table
    build_global_xor_table(magic)

    # Decode each message
    for i, msg in enumerate(messages):
        direction = msg["direction"]
        payload = msg["payload"]

        result = _decode_sent(payload) if direction == "sent" else _decode_received(payload)

        if result is not None:
            log.info("[%3d] %s", i, result)
        else:
            log.warning("[%3d] [%s] (decode failed)", i, direction.upper())

    # Print game log if present
    game_log = session.get("game_log")
    if game_log and isinstance(game_log, list) and len(game_log) > 0:
        log.info("")
        log.info("=== Game Log (%d entries) ===", len(game_log))
        for entry in game_log:
            text = entry.get("text", "")
            cat = entry.get("category", "other")
            log.info("[%s] %s", cat.upper(), text)


if __name__ == "__main__":
    main()


__all__ = [
    "main",
]
