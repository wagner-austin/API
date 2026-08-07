"""Shared payload and session builders for the viewport-analysis tests."""

from __future__ import annotations

from tankpit_bot.types.message import CapturedMessage
from tankpit_bot.types.session import CaptureSession
from tests.wire_builders import encode_wire_frame


def _make_sync_payload(xor_table: bytes) -> str:
    """Create a Sync payload.

    Args:
        xor_table: Session XOR table.

    Returns:
        Base64-encoded received frame payload.
    """
    return encode_wire_frame(0x3F, b"", xor_table)


def _make_session(messages: list[CapturedMessage], magic: str) -> CaptureSession:
    """Create a typed capture session for tests.

    Args:
        messages: Captured messages.
        magic: Session magic string.

    Returns:
        CaptureSession containing the provided messages.
    """
    return CaptureSession(
        session_id="viewport-analysis-test",
        start_timestamp_ms=1000,
        end_timestamp_ms=2000,
        base_url="https://tankpit.com/play",
        messages=messages,
        magic=magic,
        game_log=[],
        tank_names={},
    )
