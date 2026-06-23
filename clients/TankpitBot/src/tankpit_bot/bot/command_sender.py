"""Low-level command encoding and WebSocket dispatch.

Extracted from bot.base to give command dispatch its own module.
The Bot class delegates to these functions.
"""

from __future__ import annotations

from typing import Protocol

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.protocol.codec import xor_bytes
from tankpit_bot.protocol.commands import COMMAND_PREFIX
from tankpit_bot.runtime_logging import emit_wire

log = get_logger(__name__)


def xor_encode_command(xor_table: bytes | None, data: bytes) -> bytes:
    """XOR encode a framed command for wire transmission.

    Args:
        xor_table: XOR table, or None if not yet discovered.
        data: Framed command bytes (with 2-byte length header).

    Returns:
        XOR-encoded framed command bytes.
    """
    if xor_table is None or len(data) < 4:
        return data
    header = data[:2]
    prefix = data[2:3]
    payload = data[3:]
    encoded_payload = xor_bytes(xor_table, payload, offset=0)
    return header + prefix + encoded_payload


def send_command_bytes(
    cdp: CDPSessionProtocol | None,
    xor_table: bytes | None,
    data: bytes,
    cmd_name: str,
    send_ws_bytes: SendWebSocketBytesFunc,
) -> bool:
    """XOR encode and send command bytes via WebSocket.

    Args:
        cdp: CDP session, or None if not connected.
        xor_table: XOR table for encoding.
        data: Framed command bytes.
        cmd_name: Command name for logging.
        send_ws_bytes: Callback to send raw bytes over the WebSocket.

    Returns:
        True if sent, False if CDP session not available.
    """
    if cdp is None:
        log.warning("Cannot send %s: CDP session not available", cmd_name)
        return False
    if len(data) > 2 and data[2] == COMMAND_PREFIX:
        data = xor_encode_command(xor_table, data)
    send_ws_bytes(cdp, data, cmd_name)
    # Parse the action prefix out of names like "shoot(...)" so
    # smoke / bot-query can filter by ``action_kind`` without
    # parsing the message text. Names without parens (e.g.
    # ``map_open``) stand as the action_kind verbatim.
    paren = cmd_name.find("(")
    action_kind = cmd_name[:paren] if paren > 0 else cmd_name
    emit_wire("%s", cmd_name, action_kind=action_kind)
    return True


class SendWebSocketBytesFunc(Protocol):
    """Protocol for the WebSocket byte sender callback."""

    def __call__(self, cdp: CDPSessionProtocol, data: bytes, label: str) -> str: ...


__all__ = [
    "SendWebSocketBytesFunc",
    "send_command_bytes",
    "xor_encode_command",
]
