"""Tests for shared action-lab raw capture persistence."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.conftest import FakeFileSystem

from tankpit_bot.action_lab.capture import build_capture_output_path, save_capture_session
from tankpit_bot.types import CapturedMessage, decode_capture_session


def test_build_capture_output_path_replaces_existing_suffix() -> None:
    """Capture output path reuses the probe filename stem."""
    assert build_capture_output_path("teleport_probe.json") == "teleport_probe.capture_session.json"


def test_build_capture_output_path_appends_suffix_when_missing() -> None:
    """Capture output path appends the capture suffix when no extension exists."""
    assert build_capture_output_path("teleport_probe") == "teleport_probe.capture_session.json"


def test_save_capture_session_writes_replayable_session(fake_fs: FakeFileSystem) -> None:
    """Raw capture persistence writes a valid replayable capture session."""
    messages = [
        CapturedMessage(
            timestamp_ms=10,
            direction="sent",
            payload="abc",
            ws_url="wss://dorothy.tankpit.com/ws/",
        )
    ]

    capture_path = save_capture_session(
        session_id="teleport-session",
        start_timestamp_ms=1,
        end_timestamp_ms=20,
        base_url="https://tankpit.com/play",
        messages=messages,
        magic="magic",
        output_path="teleport_probe.json",
    )

    assert capture_path == "teleport_probe.capture_session.json"
    written = fake_fs.read_text(Path(capture_path))
    decoded = decode_capture_session(narrow_json_to_dict(load_json_str(written)))
    assert decoded["session_id"] == "teleport-session"
    assert decoded["messages"] == messages
    assert decoded["magic"] == "magic"
