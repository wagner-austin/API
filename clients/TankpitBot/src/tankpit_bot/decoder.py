"""Decode captured WebSocket sessions using XOR codec.

Loads a capture session JSON, extracts the magic key, builds the XOR table,
and decodes all command messages to reveal actual protocol bytes.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal, TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.protocol.codec import ProtocolCodec, load_static_key
from tankpit_bot.protocol.framing import decode_frame_header
from tankpit_bot.resources import static_key_file_path
from tankpit_bot.types import CaptureSession, decode_capture_session

log = get_logger(__name__)


class DecoderError(Exception):
    """Error during session decoding."""


class MissingMagicError(DecoderError):
    """Raised when capture session has no magic key."""


# =============================================================================
# Decoded Message Types
# =============================================================================


class DecodedCommand(TypedDict):
    """A decoded game command.

    Attributes:
        timestamp_ms: When the message was captured.
        direction: Whether sent or received.
        raw_hex: Original XOR'd bytes (hex string).
        decoded_hex: Decoded bytes after XOR (hex string).
        type_byte: The type byte from position 1.
        cmd_byte: The command byte from position 2.
        data_hex: Command data payload (hex string, may be empty).
    """

    timestamp_ms: int
    direction: Literal["sent", "received"]
    raw_hex: str
    decoded_hex: str
    type_byte: int
    cmd_byte: int
    data_hex: str


class DecodedLobbyMessage(TypedDict):
    """A decoded lobby/text message.

    Attributes:
        timestamp_ms: When the message was captured.
        direction: Whether sent or received.
        prefix: Message prefix character.
        text: Message text content.
    """

    timestamp_ms: int
    direction: Literal["sent", "received"]
    prefix: str
    text: str


def encode_decoded_command(cmd: DecodedCommand) -> JSONObject:
    """Encode DecodedCommand to JSON-serializable dict.

    Args:
        cmd: DecodedCommand to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "timestamp_ms": cmd["timestamp_ms"],
        "direction": cmd["direction"],
        "raw_hex": cmd["raw_hex"],
        "decoded_hex": cmd["decoded_hex"],
        "type_byte": cmd["type_byte"],
        "cmd_byte": cmd["cmd_byte"],
        "data_hex": cmd["data_hex"],
    }


def decode_decoded_command(data: JSONObject) -> DecodedCommand:
    """Decode DecodedCommand from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated DecodedCommand.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    direction = require_str(data, "direction")
    if direction not in ("sent", "received"):
        raise JSONTypeError(f"Invalid direction: {direction}")

    direction_literal: Literal["sent", "received"] = "sent" if direction == "sent" else "received"

    return DecodedCommand(
        timestamp_ms=require_int(data, "timestamp_ms"),
        direction=direction_literal,
        raw_hex=require_str(data, "raw_hex"),
        decoded_hex=require_str(data, "decoded_hex"),
        type_byte=require_int(data, "type_byte"),
        cmd_byte=require_int(data, "cmd_byte"),
        data_hex=require_str(data, "data_hex"),
    )


def encode_decoded_lobby_message(msg: DecodedLobbyMessage) -> JSONObject:
    """Encode DecodedLobbyMessage to JSON-serializable dict.

    Args:
        msg: DecodedLobbyMessage to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "timestamp_ms": msg["timestamp_ms"],
        "direction": msg["direction"],
        "prefix": msg["prefix"],
        "text": msg["text"],
    }


def decode_decoded_lobby_message(data: JSONObject) -> DecodedLobbyMessage:
    """Decode DecodedLobbyMessage from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated DecodedLobbyMessage.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    direction = require_str(data, "direction")
    if direction not in ("sent", "received"):
        raise JSONTypeError(f"Invalid direction: {direction}")

    direction_literal: Literal["sent", "received"] = "sent" if direction == "sent" else "received"

    return DecodedLobbyMessage(
        timestamp_ms=require_int(data, "timestamp_ms"),
        direction=direction_literal,
        prefix=require_str(data, "prefix"),
        text=require_str(data, "text"),
    )


# =============================================================================
# Session Decoder
# =============================================================================


class SessionDecoder:
    """Decodes a captured session using XOR codec."""

    def __init__(self, session: CaptureSession, codec: ProtocolCodec) -> None:
        """Initialize decoder.

        Args:
            session: Captured session to decode.
            codec: XOR codec built from static key + magic.
        """
        self._session = session
        self._codec = codec
        self._commands: list[DecodedCommand] = []
        self._lobby_messages: list[DecodedLobbyMessage] = []

    def decode_all(self) -> None:
        """Decode all messages in the session."""
        self._commands = []
        self._lobby_messages = []

        for msg in self._session["messages"]:
            self._decode_message(
                msg["timestamp_ms"],
                msg["direction"],
                msg["payload"],
            )

    def _decode_message(
        self,
        timestamp_ms: int,
        direction: Literal["sent", "received"],
        payload: str,
    ) -> None:
        """Decode a single message.

        Args:
            timestamp_ms: Message timestamp.
            direction: Message direction.
            payload: Base64-encoded message payload.
        """
        raw_bytes = base64.b64decode(payload)

        # Need at least 2 bytes for frame header
        if len(raw_bytes) < 2:
            return

        # Parse frame header to get body
        body_length = decode_frame_header(raw_bytes)

        if len(raw_bytes) < 2 + body_length:
            return

        body = raw_bytes[2 : 2 + body_length]
        if len(body) == 0:
            return

        # Check first byte to determine message type
        first_byte = body[0]

        if first_byte == ord("!"):
            # Game command - decode with XOR
            self._decode_command(timestamp_ms, direction, body)
        elif first_byte == ord("."):
            # State update - binary, skip for now
            pass
        elif first_byte in (ord("%"), ord("+"), ord("*"), ord("="), ord("$"), ord("-")):
            # Lobby message - text
            self._decode_lobby_message(timestamp_ms, direction, body)

    def _decode_command(
        self,
        timestamp_ms: int,
        direction: Literal["sent", "received"],
        body: bytes,
    ) -> None:
        """Decode a game command.

        Args:
            timestamp_ms: Message timestamp.
            direction: Message direction.
            body: Command body (starts with '!').
        """
        if len(body) < 3:
            return

        raw_hex = body.hex()

        # The '!' prefix is NOT XOR encoded, only the bytes after it.
        # XOR decode body[1:] starting at offset 0 of the XOR table.
        decoded_payload = self._codec.decode(body[1:], offset=0)
        decoded = body[0:1] + decoded_payload
        decoded_hex = decoded.hex()

        # Extract command parts from decoded bytes
        # Format: '!' + type_byte + cmd_byte + data
        type_byte = decoded_payload[0]
        cmd_byte = decoded_payload[1]
        data_hex = decoded_payload[2:].hex() if len(decoded_payload) > 2 else ""

        cmd = DecodedCommand(
            timestamp_ms=timestamp_ms,
            direction=direction,
            raw_hex=raw_hex,
            decoded_hex=decoded_hex,
            type_byte=type_byte,
            cmd_byte=cmd_byte,
            data_hex=data_hex,
        )
        self._commands.append(cmd)

    def _decode_lobby_message(
        self,
        timestamp_ms: int,
        direction: Literal["sent", "received"],
        body: bytes,
    ) -> None:
        """Decode a lobby/text message.

        Args:
            timestamp_ms: Message timestamp.
            direction: Message direction.
            body: Message body.
        """
        text = body.decode("utf-8", errors="replace")
        prefix = text[0] if len(text) > 0 else ""
        content = text[1:] if len(text) > 1 else ""

        msg = DecodedLobbyMessage(
            timestamp_ms=timestamp_ms,
            direction=direction,
            prefix=prefix,
            text=content,
        )
        self._lobby_messages.append(msg)

    @property
    def commands(self) -> list[DecodedCommand]:
        """Get decoded commands."""
        return self._commands

    @property
    def lobby_messages(self) -> list[DecodedLobbyMessage]:
        """Get decoded lobby messages."""
        return self._lobby_messages


def load_and_decode_session(
    session_path: Path,
    static_key_path: Path | None = None,
) -> SessionDecoder:
    """Load a capture session and decode it.

    Args:
        session_path: Path to capture session JSON.
        static_key_path: Path to static XOR key file. Defaults to bundled key.

    Returns:
        SessionDecoder with decoded messages.

    Raises:
        MissingMagicError: If session has no magic key.
        FileNotFoundError: If files don't exist.
    """
    if static_key_path is None:
        static_key_path = static_key_file_path()

    # Load session
    session_text = _test_hooks.read_text(session_path)
    session_json = narrow_json_to_dict(load_json_str(session_text))
    session = decode_capture_session(session_json)

    # Check for magic
    magic = session["magic"]
    if magic is None:
        raise MissingMagicError("Capture session has no magic key")

    # Load static key and build codec
    static_key = load_static_key(static_key_path)
    codec = ProtocolCodec(static_key, magic)

    # Decode
    decoder = SessionDecoder(session, codec)
    decoder.decode_all()

    return decoder


def main() -> None:
    """Entry point for tankpit-decode command."""
    from dotenv import load_dotenv
    from platform_core.rich_logging import setup_rich_logging

    load_dotenv()
    setup_rich_logging(level="INFO")

    # Get session path from env or default
    session_path_str = _test_hooks.get_env("TANKPIT_OUTPUT") or "capture_session.json"
    session_path = Path(session_path_str)

    log.info("Loading session from %s", session_path)

    decoder = load_and_decode_session(session_path)

    # Print summary
    log.info("Decoded %d commands", len(decoder.commands))
    log.info("Decoded %d lobby messages", len(decoder.lobby_messages))

    # Print commands grouped by type_byte and cmd_byte
    if len(decoder.commands) > 0:
        log.info("Commands by type:")
        type_bytes: set[int] = set()
        for cmd in decoder.commands:
            type_bytes.add(cmd["type_byte"])

        for tb in sorted(type_bytes):
            cmds_with_type = [c for c in decoder.commands if c["type_byte"] == tb]
            log.info("  type_byte=0x%02x: %d commands", tb, len(cmds_with_type))

            # Group by cmd_byte
            cmd_bytes: set[int] = set()
            for c in cmds_with_type:
                cmd_bytes.add(c["cmd_byte"])

            for cb in sorted(cmd_bytes):
                cmds_with_cb = [c for c in cmds_with_type if c["cmd_byte"] == cb]
                example = cmds_with_cb[0]
                log.info(
                    "    cmd_byte=0x%02x: %d times, data_len=%d, example=%s",
                    cb,
                    len(cmds_with_cb),
                    len(example["data_hex"]) // 2,
                    example["decoded_hex"][:20],
                )


__all__ = [
    "DecodedCommand",
    "DecodedLobbyMessage",
    "DecoderError",
    "MissingMagicError",
    "SessionDecoder",
    "decode_decoded_command",
    "decode_decoded_lobby_message",
    "encode_decoded_command",
    "encode_decoded_lobby_message",
    "load_and_decode_session",
    "main",
]
