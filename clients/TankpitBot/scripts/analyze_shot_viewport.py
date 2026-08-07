"""Correlate sent shoot commands with the latest raw 0x5A viewport update.

Usage: poetry run python -m scripts.analyze_shot_viewport [session.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from scripts import _test_hooks
from tankpit_bot.capture.shot_viewport_correlation import (
    analyze_shot_viewport_correlation,
    format_shot_viewport_correlation,
)
from tankpit_bot.capture.xor import XorStaticKeyUnavailableError
from tankpit_bot.types import decode_capture_session


def _load_session(session_path: Path) -> str:
    """Load raw capture session text from disk.

    Args:
        session_path: Path to the capture session JSON file.

    Returns:
        Raw session JSON text.

    Raises:
        FileNotFoundError: If the session file does not exist.
    """
    if not _test_hooks.path_exists(session_path):
        raise FileNotFoundError(f"File not found: {session_path}")
    return _test_hooks.read_text(session_path)


def main() -> None:
    """Correlate sent shoot commands with raw viewport entities."""
    _test_hooks.setup_rich_logging(level="INFO")

    session_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("capture_session.json")

    try:
        session_text = _load_session(session_path)
    except FileNotFoundError as exc:
        sys.stdout.write(f"{exc}\n")
        raise SystemExit(1) from exc

    session_json = narrow_json_to_dict(load_json_str(session_text))
    session = decode_capture_session(session_json)

    try:
        result = analyze_shot_viewport_correlation(session)
    except (ValueError, XorStaticKeyUnavailableError) as exc:
        sys.stdout.write(f"{exc}\n")
        raise SystemExit(1) from exc

    sys.stdout.write(format_shot_viewport_correlation(result))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()


__all__ = ["main"]
