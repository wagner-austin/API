"""Shared raw capture persistence for action-lab probes."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import dump_json_str

from tankpit_bot import _test_hooks
from tankpit_bot.types import CapturedMessage, CaptureSession, encode_capture_session


def build_capture_output_path(output_path: str) -> str:
    """Build the raw capture output path for a probe result file.

    Args:
        output_path: Structured probe JSON output path.

    Returns:
        Sibling path for the replayable raw capture session.
    """
    path = Path(output_path)
    if path.suffix:
        return str(path.with_suffix(".capture_session.json"))
    return f"{output_path}.capture_session.json"


def save_capture_session(
    *,
    session_id: str,
    start_timestamp_ms: int,
    end_timestamp_ms: int,
    base_url: str,
    messages: list[CapturedMessage],
    magic: str | None,
    output_path: str,
) -> str:
    """Persist a replayable raw capture session for an action-lab probe.

    Args:
        session_id: Probe session identifier.
        start_timestamp_ms: Probe start timestamp.
        end_timestamp_ms: Probe end timestamp.
        base_url: Target URL used by the probe.
        messages: Raw captured websocket messages.
        magic: Captured XOR magic key, if available.
        output_path: Structured probe JSON output path.

    Returns:
        Capture session output path.
    """
    capture_output_path = build_capture_output_path(output_path)
    session = CaptureSession(
        session_id=session_id,
        start_timestamp_ms=start_timestamp_ms,
        end_timestamp_ms=end_timestamp_ms,
        base_url=base_url,
        messages=messages,
        magic=magic,
        game_log=[],
        tank_names={},
    )
    encoded = encode_capture_session(session)
    json_str = dump_json_str(encoded, compact=False, indent=2)
    _test_hooks.write_text(Path(capture_output_path), json_str)
    return capture_output_path


__all__ = [
    "build_capture_output_path",
    "save_capture_session",
]
