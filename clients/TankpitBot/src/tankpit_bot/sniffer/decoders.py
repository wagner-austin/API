"""Message decoding functions for WebSocket traffic.

This module provides functions to decode raw WebSocket message payloads
into human-readable strings for logging and analysis.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import protocol
from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.capture.xor import build_session_xor_table, decode_base64_safe, xor_decode_body
from tankpit_bot.parser import is_room_info_text
from tankpit_bot.protocol.constants import MSG_PROMOTION, RANK_NAMES
from tankpit_bot.protocol.decoders import try_decode_plaintext_ack
from tankpit_bot.sniffer.constants import MSG_MIN_LENGTHS, TEXT_MESSAGE_TYPES
from tankpit_bot.sniffer.formatters import format_decoded_message
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update

log = get_logger(__name__)

# Binary Promotion (Rf) carries exactly 2 XOR-decoded payload bytes
# (new_rank, was_promoted) -> body length 3. Text WorldInfo/ROOM_LIST
# (also 0x2B '+') is far longer (pipe-delimited fields). This is the
# only place in the wire grammar where a single byte type covers both
# a text and a binary path; we disambiguate by length.
_BINARY_PROMOTION_BODY_LEN = 3


def _is_text_route(msg_type: int, body: bytes) -> bool:
    """Decide whether a body should take the text-log path.

    The 0x2B '+' byte is dual-use: text WorldInfo / ROOM_LIST at lobby
    vs. binary Promotion (Rf) during gameplay. Wire length is the only
    field that uniquely separates them (3 bytes for Rf, far more for
    text). Every other entry in ``TEXT_MESSAGE_TYPES`` is text-only.

    Args:
        msg_type: First byte of the message body.
        body: Full body bytes including ``msg_type``.

    Returns:
        True if the body should be formatted/logged as text.
    """
    if msg_type not in TEXT_MESSAGE_TYPES:
        return False
    return not (msg_type == MSG_PROMOTION and len(body) == _BINARY_PROMOTION_BODY_LEN)


_PROTOCOL_FRAME_LOGGING_ENABLED = True


def set_protocol_frame_logging(enabled: bool) -> None:
    """Enable or disable per-message protocol frame logging.

    Args:
        enabled: True to log decoded protocol messages, False to suppress them.
    """
    global _PROTOCOL_FRAME_LOGGING_ENABLED
    _PROTOCOL_FRAME_LOGGING_ENABLED = enabled


def _log_protocol_line(message: str) -> None:
    """Log a decoded protocol line when protocol frame logging is enabled.

    Args:
        message: Formatted protocol log line.
    """
    if _PROTOCOL_FRAME_LOGGING_ENABLED:
        log.info(message)


def process_received_message(ws: WorldService, payload: str, xor_table: bytes) -> None:
    """Decode, log, and dispatch received messages to world state.

    A single WebSocket frame can contain multiple logical messages, each
    with a 2-byte little-endian length prefix. The split is
    :func:`~tankpit_bot.capture.frames.split_payload_frames`; this used
    to re-derive it inline and drop a torn tail with a silent ``break``
    ([[session-state-deglobalisation]]).

    Args:
        ws: The SESSION's world service. Passed in rather than reached
            through a module singleton so two sessions cannot dispatch
            into each other's world ([[session-state-deglobalisation]]
            step 8) — the same reason ``xor_table`` is a parameter.
        payload: Base64-encoded WebSocket frame payload.
        xor_table: The SESSION's XOR table. Passed in rather than read
            from a module global so two sessions cannot decode each
            other's frames ([[session-state-deglobalisation]] step 1).

    Raises:
        FramingError: If the live payload is corrupt. Measured over the
            whole archive (230,323 received payloads, 407 sessions):
            zero. The live socket delivers whole frames, so this is a
            real fault rather than a routine condition to absorb.
    """
    for body in split_payload_frames(payload):
        _process_single_message(ws, body, xor_table)


def _process_single_message(ws: WorldService, body: bytes, xor_table: bytes) -> None:
    """Process a single logical message (after frame splitting).

    Args:
        ws: The SESSION's world service to dispatch into.
        body: Non-empty message body bytes (without length prefix).
            The frame parser guarantees msg_len > 0 before calling.
        xor_table: The SESSION's XOR table.
    """
    msg_type = body[0]

    # Plaintext toggle acks (un-XORed two-byte echoes) — discriminated
    # before the XOR route because their letters are overloaded with
    # binary frames. Acks carry no world state; log only.
    ack = try_decode_plaintext_ack(body)
    if ack is not None:
        _log_protocol_line(f"[RECEIVED] {format_decoded_message(msg_type, ack)}")
        return

    # Text messages — log only.
    if _is_text_route(msg_type, body):
        text = body.decode("utf-8", errors="replace")
        result = decode_text_message(ws, text, len(body), "RECEIVED", body)
        _log_protocol_line(result)
        return

    # Binary messages — XOR decode, log, and dispatch
    decoded_data = xor_decode_body(body, xor_table, offset=1)
    if len(decoded_data) == 0:
        return

    # Every type in MSG_MIN_LENGTHS has a top-level decoder, so
    # decode_message cannot answer "unknown type" for anything the table
    # admits. That was NOT true until 2026-08-12 -- the table listed the
    # container-only subtypes 0x45 and 0x4B, which decode_message has no
    # case for -- and it is now enforced by
    # test_every_declared_type_is_decodable_at_top_level rather than
    # asserted here.
    min_len = MSG_MIN_LENGTHS.get(msg_type)
    if min_len is not None and len(decoded_data) >= min_len:
        binary_decoded = protocol.decode_message(msg_type, decoded_data)
        _log_protocol_line(f"[RECEIVED] {format_decoded_message(msg_type, binary_decoded)}")
        dispatch_world_state_update(ws, binary_decoded)
        return
    msg_char = chr(msg_type) if 32 <= msg_type < 127 else "?"
    _log_protocol_line(f"[RECEIVED] type=0x{msg_type:02X} '{msg_char}' len={len(body)}")


def try_decode_binary(ws: WorldService, msg_type: int, data: bytes, raw_body: bytes) -> str:
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

    # Every type in MSG_MIN_LENGTHS has a top-level decoder, so
    # decode_message cannot answer "unknown type" for anything the table
    # admits. That was NOT true until 2026-08-12 -- the table listed the
    # container-only subtypes 0x45 and 0x4B, which decode_message has no
    # case for -- and it is now enforced by
    # test_every_declared_type_is_decodable_at_top_level rather than
    # asserted here.
    binary_decoded = protocol.decode_message(msg_type, data)

    dispatch_world_state_update(ws, binary_decoded)

    return format_decoded_message(msg_type, binary_decoded)


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


def decode_text_message(
    ws: WorldService, text: str, body_len: int, tag: str, body: bytes | None = None
) -> str:
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
        return decode_plus_message(ws, text, tag)
    if text.startswith("*"):
        return f"[{tag}] SELECT: room={text[1:]}"
    if text.startswith("="):
        return decode_join_confirm(ws, text, tag)
    if text.startswith("$"):
        return f"[{tag}] RESPONSE: {text}"
    if text.startswith(".") and body is not None:
        return decode_state_message(body, tag)
    if text.startswith("."):
        return f"[{tag}] STATE: len={body_len} bytes"
    # Unknown - show first 40 chars
    preview = text[:40].replace("\n", " ")
    return f"[{tag}] ???: {preview}..."


def decode_message(ws: WorldService, payload: str, direction: str, magic: str | None) -> str:
    """Decode a WebSocket message payload for display.

    Args:
        ws: The session's world service; room beliefs land here.
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
    return decode_text_message(ws, text, len(body), tag, body)


def decode_plus_message(ws: WorldService, text: str, tag: str) -> str:
    """Decode a '+' prefixed message (ROOM_LIST or ACTION).

    Room list format: +room_id|name|field_id|modes|default_troop|mode_code|image|year
    Example: +2|World (Meltdown)|24|1,1,1,0,1,0,0|2|n|field24.gif|2026

    Args:
        text: Text starting with '+'.
        tag: Direction tag.

    Returns:
        Decoded message string.
    """
    room_text = text[1:]
    parts = room_text.split("|")
    if is_room_info_text(room_text):
        room_id = parts[0]
        name = parts[1]
        ws.register_room_image(room_id, parts[6])
        return f"[{tag}] ROOM_LIST: room={room_id} name={name}"
    # Action message with coords
    room_id = parts[0] if len(parts) > 0 else "?"
    coords = f"{parts[2]},{parts[3]}" if len(parts) >= 4 else "?"
    return f"[{tag}] ACTION: room={room_id} coords={coords}"


def decode_join_confirm(ws: WorldService, text: str, tag: str) -> str:
    """Decode a '=' prefixed JOIN_CONFIRM message.

    Format: =room|date|name|rank|f5|f6|f7|f8
    Example: =2|Sep. 25, 2012|Yuppler|4|9|9|9|10

    Rank values: 0=recruit, 1=private, 2=corporal, 3=sergeant,
                 4=lieutenant, 5=captain, 6=major, 7=colonel, 8=general

    Fields 5-8 are UNIDENTIFIED and logged verbatim. They were named
    ``eq1..eq4`` here and ``equipment`` in
    :class:`~tankpit_bot.types.message.JoinConfirmDict`, but nothing
    ever verified that. The competing reading comes from the client
    itself: ``tpclient.js`` builds the lobby stats panel as ``Rank:``
    followed immediately by ``Active Forces / Orange: / Purple: /
    Blue: / Red:`` from four variables, which would make these
    per-color PLAYER COUNTS for the room rather than inventory.
    Logging them raw is what lets the two readings be told apart
    against ``tankpit.com/api/active_games``, which reports the same
    rooms' live populations without a login.

    Args:
        text: Text starting with '='.
        tag: Direction tag.

    Returns:
        Decoded message string.
    """
    parts = text.split("|")
    room_id = parts[0][1:] if len(parts) > 0 else "?"
    if room_id:
        ws.set_selected_room(room_id)
    tank_name = parts[2] if len(parts) > 2 else "?"
    rank_num = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else -1
    rank_str = RANK_NAMES[rank_num] if 0 <= rank_num < len(RANK_NAMES) else f"rank{rank_num}"
    trailing = ",".join(parts[4:8]) if len(parts) > 4 else "-"
    return f"[{tag}] JOIN_CONFIRM: room={room_id} tank={tank_name} {rank_str} f5-8={trailing}"


def decode_command(body: bytes, tag: str, magic: str | None) -> str:
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

    # XOR decrypt if magic is available. This block used to inline the
    # WHOLE cipher — its own copy of the key path, its own file read,
    # its own table build, and its own offset-1 XOR loop — and fall
    # silently through to the hex dump when the key was missing. It was
    # invisible to a by-name sweep for the shared helpers precisely
    # because it named none of them ([[session-state-deglobalisation]]).
    if magic:
        decrypted = bytearray(body)
        decrypted[1:] = xor_decode_body(body, build_session_xor_table(magic), offset=1)

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

    # Fallback to hex when the session has no magic at all.
    return f"[{tag}] CMD: ! {body.hex()}"


__all__ = [
    "decode_8byte_state",
    "decode_command",
    "decode_join_confirm",
    "decode_message",
    "decode_plus_message",
    "decode_state_message",
    "decode_text_message",
    "process_received_message",
    "try_decode_binary",
]
