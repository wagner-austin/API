"""Helpers for loading checked-in replay fixtures."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot.types import CaptureSession, decode_capture_session

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def get_replay_fixture_path(filename: str) -> Path:
    """Return the absolute path to a replay fixture file.

    Args:
        filename: Fixture filename under ``tests/replay/fixtures``.

    Returns:
        Absolute path to the fixture file.

    Raises:
        FileNotFoundError: If the fixture file does not exist.
    """
    path = _FIXTURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_capture_fixture(filename: str) -> CaptureSession:
    """Load and decode a checked-in capture session fixture.

    Args:
        filename: Fixture filename under ``tests/replay/fixtures``.

    Returns:
        Validated capture session.

    Raises:
        FileNotFoundError: If the fixture file does not exist.
        ValueError: If the fixture JSON is not an object.
    """
    path = get_replay_fixture_path(filename)
    text = path.read_text(encoding="utf-8")
    raw_session = narrow_json_to_dict(load_json_str(text))
    return decode_capture_session(raw_session)


__all__ = [
    "get_replay_fixture_path",
    "load_capture_fixture",
]
