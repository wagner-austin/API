"""Message decoding functions for WebSocket traffic.

This module provides functions to decode raw WebSocket message payloads
into human-readable strings for logging and analysis.
"""

from __future__ import annotations

import base64
from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks, protocol
from tankpit_bot.capture import decode_base64_safe
from tankpit_bot.sniffer.constants import MSG_MIN_LENGTHS, TEXT_MESSAGE_TYPES
from tankpit_bot.sniffer.formatters import format_decoded_message
from tankpit_bot.sniffer.world_state import dispatch_world_state_update
from tankpit_bot.sniffer.xor import xor_decode

log = get_logger(__name__)


def try_decode_received_text(payload: str) -> str | None:
    """Try to decode a received text message payload.

    Pure function that returns decoded string or None. Does not log.

    Args:
        payload: Base64-encoded message payload.

    Returns:
        Decoded message string, or None if not a valid text message.
    """
    # Validate base64 - must be valid characters and proper length
    if not payload or len(payload) % 4 != 0:
        return None
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    if not all(c in valid_chars for c in payload):
        return None

    data = base64.b64decode(payload)
    if len(data) < 2:
        return None

    body = data[2:]
    if len(body) == 0:
        return None

    # Only process text message types
    if body[0] not in TEXT_MESSAGE_TYPES:
        return None

    text = body.decode("utf-8", errors="replace")
    return decode_text_message(text, len(body), "RECEIVED", body)


def decode_received_text_message(payload: str) -> None:
    """Decode and log received text messages (JOIN_CONFIRM, ROOM_LIST, etc.).

    Args:
        payload: Base64-encoded message payload.
    """
    result = try_decode_received_text(payload)
    if result is not None:
        log.info(result)


def try_decode_received(payload: str) -> str | None:
    """Try to decode a received message and return formatted result.

    Pure function that returns the decoded message string or None. Does not log.

    Args:
        payload: Base64-encoded message payload.

    Returns:
        Decoded message string, or None if payload is invalid/empty.
    """
    data = decode_base64_safe(payload)
    if data is None or len(data) < 3:
        return None

    # body is guaranteed non-empty since len(data) >= 3 means len(body) >= 1
    body = data[2:]
    msg_type = body[0]

    # Text messages (not XOR encoded)
    if msg_type in TEXT_MESSAGE_TYPES:
        text = body.decode("utf-8", errors="replace")
        return decode_text_message(text, len(body), "RECEIVED", body)

    # Binary messages - XOR decode and use protocol module
    decoded_data = xor_decode(body)
    if len(decoded_data) == 0:
        return f"[RECEIVED] EMPTY: type=0x{msg_type:02X}"

    # All messages go through protocol.decode_message (handles 0x2E routing internally)
    return "[RECEIVED] " + try_decode_binary(msg_type, decoded_data, body)


def process_received_message(payload: str) -> None:
    """Decode and log ALL received messages using protocol module.

    Args:
        payload: Base64-encoded message payload.
    """
    result = try_decode_received(payload)
    if result is not None:
        log.info(result)


def try_decode_binary(msg_type: int, data: bytes, raw_body: bytes) -> str:
    """Try to decode a binary message and return formatted result string.

    Pure function that returns the formatted decode result. Does not log.

    Args:
        msg_type: Message type byte.
        data: XOR-decoded message data (without msg_type byte).
        raw_body: Original raw body for length reporting.

    Returns:
        Formatted decode result string (UNKNOWN/SHORT/UNIMPL/decoded message).
    """
    msg_char = chr(msg_type) if 32 <= msg_type < 127 else "?"
    hex_preview = data[:20].hex() + "..." if len(data) > 20 else data.hex()

    # Check if type is known and data meets minimum length
    min_len = MSG_MIN_LENGTHS.get(msg_type)
    if min_len is None:
        # Unknown type - show debug info
        return f"UNKNOWN 0x{msg_type:02X} '{msg_char}' len={len(raw_body)} data={hex_preview}"

    if len(data) < min_len:
        # Data too short for this type
        return (
            f"SHORT 0x{msg_type:02X} '{msg_char}' need={min_len} got={len(data)} data={hex_preview}"
        )

    # Decode using protocol module - for binary messages only
    # Text messages are handled separately in decode_text_message
    # All types in MSG_MIN_LENGTHS have corresponding decoder implementations,
    # so try_decode_binary_message will return a value (not None) for any type
    # that passes the min_len check above.
    binary_decoded = protocol.try_decode_binary_message(msg_type, data)
    assert binary_decoded is not None, f"Missing decoder for type 0x{msg_type:02X}"

    # Update world state from decoded message
    dispatch_world_state_update(binary_decoded)

    return format_decoded_message(msg_type, binary_decoded)


def decode_and_log_binary(msg_type: int, data: bytes, _type_label: str, raw_body: bytes) -> None:
    """Decode binary message using protocol module and log it.

    Args:
        msg_type: Message type byte.
        data: XOR-decoded message data (without msg_type byte).
        _type_label: Unused parameter (kept for API compatibility).
        raw_body: Original raw body for length reporting.
    """
    result = try_decode_binary(msg_type, data, raw_body)
    log.info("[RECEIVED] %s", result)


def decode_8byte_state(body: bytes, tag: str) -> str:
    """Decode 8-byte state message by subtype.

    Args:
        body: Raw body bytes.
        tag: Direction tag (SENT/RECEIVED).

    Returns:
        Decoded state message string.
    """
    subtype = body[1]
    if subtype == 0x49:
        return f"[{tag}] ITEM_PICKUP: {body.hex()}"
    if subtype == 0x67:
        return f"[{tag}] GAME_STATE: {body.hex()}"
    return f"[{tag}] MSG_8B: sub=0x{subtype:02x} {body.hex()}"


def decode_state_message(body: bytes, tag: str) -> str:
    """Decode a '.' prefixed state message based on length/pattern.

    State message types by length:
    - 2-3 bytes: Heartbeat/sync
    - 4-8 bytes: Entity position refs
    - 12 bytes: Hit confirmation (after shots)
    - 14 bytes: Tank status with fuel
    - 17-30 bytes: Entity updates
    - 673+ bytes: Map data

    Args:
        body: Raw body bytes starting with '.'.
        tag: Direction tag (SENT/RECEIVED).

    Returns:
        Human-readable decoded state string.
    """
    length = len(body)

    if length <= 3:
        return f"[{tag}] SYNC: {body.hex()}"

    if length > 500:
        return f"[{tag}] MAP_DATA: len={length}"

    if length == 12:
        return f"[{tag}] HIT: {body.hex()}"

    if length == 8:
        return decode_8byte_state(body, tag)

    if 14 <= length <= 16:
        return f"[{tag}] STATE: sub=0x{body[1]:02x} len={length} hex={body.hex()}"

    if length == 17 and body[1] == 0x10:
        raw_p15 = int.from_bytes(body[15:17], "little")
        return f"[{tag}] FUEL_RAW: p15={raw_p15} hex={body.hex()}"

    if 17 <= length <= 30:
        return f"[{tag}] ENTITY: sub=0x{body[1]:02x} len={length} hex={body.hex()}"

    if 4 <= length <= 11:
        return f"[{tag}] POS: len={length} hex={body.hex()}"

    return f"[{tag}] UPDATE: len={length} hex={body[:20].hex()}..."


def decode_text_message(text: str, body_len: int, tag: str, body: bytes | None = None) -> str:
    """Decode a text-based protocol message.

    Args:
        text: Decoded text body.
        body_len: Original body length in bytes.
        tag: Direction tag (SENT/RECEIVED).
        body: Raw body bytes for binary state messages.

    Returns:
        Human-readable decoded message string.
    """
    if text == "-":
        return f"[{tag}] QUIT: -"
    if text.startswith("%AUTH"):
        return f"[{tag}] AUTH: {text[:60]}..."
    if text.startswith("+") and "|" in text:
        return decode_plus_message(text, tag)
    if text.startswith("*"):
        return f"[{tag}] SELECT: room={text[1:]}"
    if text.startswith("="):
        return decode_join_confirm(text, tag)
    if text.startswith("$"):
        return f"[{tag}] RESPONSE: {text}"
    if text.startswith(".") and body is not None:
        return decode_state_message(body, tag)
    if text.startswith("."):
        return f"[{tag}] STATE: len={body_len} bytes"
    # Unknown - show first 40 chars
    preview = text[:40].replace("\n", " ")
    return f"[{tag}] ???: {preview}..."


def decode_message(payload: str, direction: str, magic: str | None = None) -> str:
    """Decode a WebSocket message payload for display.

    Args:
        payload: Base64-encoded message payload.
        direction: 'sent' or 'received'.
        magic: Captured XOR magic key.

    Returns:
        Human-readable decoded message string.
    """
    tag = direction.upper()
    data = decode_base64_safe(payload)
    if data is None:
        return f"[{tag}] (invalid base64)"

    if len(data) < 2:
        return f"[{tag}] (too short: {data.hex()})"

    # Header is 2-byte little-endian length, body follows
    body = data[2:]

    # Handle XOR commands (starting with '!')
    if len(body) > 0 and body[0] == 0x21:  # 0x21 is '!'
        return decode_command(body, tag, magic)

    text = body.decode("utf-8", errors="replace")
    return decode_text_message(text, len(body), tag, body)


def decode_plus_message(text: str, tag: str) -> str:
    """Decode a '+' prefixed message (ROOM_LIST or ACTION).

    Args:
        text: Text starting with '+'.
        tag: Direction tag.

    Returns:
        Decoded message string.
    """
    parts = text.split("|")
    if len(parts) >= 3 and len(parts[0]) > 1 and parts[0][1:].isdigit():
        room_id = parts[0][1:]
        name = parts[1] if len(parts) > 1 else "?"
        return f"[{tag}] ROOM_LIST: room={room_id} name={name}"
    # Action message with coords
    room_id = parts[0][1:] if len(parts) > 0 else "?"
    coords = f"{parts[2]},{parts[3]}" if len(parts) >= 4 else "?"
    return f"[{tag}] ACTION: room={room_id} coords={coords}"


def decode_join_confirm(text: str, tag: str) -> str:
    """Decode a '=' prefixed JOIN_CONFIRM message.

    Format: =room|date|name|rank|eq1|eq2|eq3|eq4
    Example: =2|Sep. 25, 2012|Yuppler|4|9|9|9|10

    Rank values: 0=recruit, 1=private, 2=corporal, 3=sergeant,
                 4=lieutenant, 5=captain, 6=major, 7=general

    Args:
        text: Text starting with '='.
        tag: Direction tag.

    Returns:
        Decoded message string.
    """
    rank_names = [
        "recruit",
        "private",
        "corporal",
        "sergeant",
        "lieutenant",
        "captain",
        "major",
        "general",
    ]
    parts = text.split("|")
    room_id = parts[0][1:] if len(parts) > 0 else "?"
    tank_name = parts[2] if len(parts) > 2 else "?"
    rank_num = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else -1
    rank_str = rank_names[rank_num] if 0 <= rank_num < 8 else f"rank{rank_num}"
    return f"[{tag}] JOIN_CONFIRM: room={room_id} tank={tank_name} {rank_str}"


def decode_command(body: bytes, tag: str, magic: str | None = None) -> str:
    """Decode a '!' prefixed command message.

    Args:
        body: Raw body bytes starting with '!'.
        tag: Direction tag.
        magic: XOR magic key for decryption.

    Returns:
        Decoded command string.
    """
    if len(body) < 3:
        return f"[{tag}] CMD: ! (too short: {body.hex()})"

    # XOR decrypt if magic is available
    if magic:
        # Load static key (assuming same directory as this file)
        static_key_path = Path(__file__).parent.parent.parent.parent / "xor_static_key.txt"
        if _test_hooks.path_exists(static_key_path):
            static_key = _test_hooks.read_text(static_key_path).strip()
            # Build table
            table = bytearray(len(static_key))
            for i in range(len(static_key)):
                table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])

            # Decrypt
            decrypted = bytearray(len(body))
            decrypted[0] = body[0]  # '!'
            for i in range(1, len(body)):
                decrypted[i] = body[i] ^ table[i - 1]

            cmd_type = decrypted[1]
            cmd_id = decrypted[2]

            # Decode movement commands (type=4) with coordinates
            if cmd_type == 4 and len(decrypted) >= 5:
                x = decrypted[3]
                y = decrypted[4]
                cmd_name = {112: "MOVE", 106: "PICKUP", 116: "TELEPORT"}.get(cmd_id, "?")
                return f"[{tag}] {cmd_name}: ({x}, {y})"

            # Decode shoot commands (type=6) with target
            if cmd_type == 6 and len(decrypted) >= 5:
                x = decrypted[3]
                y = decrypted[4]
                return f"[{tag}] SHOOT: ({x}, {y})"

            return f"[{tag}] CMD: ! type={cmd_type} id={cmd_id}"

    # Fallback to hex if no magic or decrypt failed
    return f"[{tag}] CMD: ! {body.hex()}"


__all__ = [
    "decode_8byte_state",
    "decode_and_log_binary",
    "decode_command",
    "decode_join_confirm",
    "decode_message",
    "decode_plus_message",
    "decode_received_text_message",
    "decode_state_message",
    "decode_text_message",
    "process_received_message",
    "try_decode_binary",
    "try_decode_received",
    "try_decode_received_text",
]
