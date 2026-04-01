"""Analyze viewport semantics from a captured session.

Usage: poetry run python -m scripts.analyze_viewport [session.json]
"""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from scripts import _test_hooks
from tankpit_bot.capture import (
    analyze_capture_session,
    build_xor_table,
    format_viewport_analysis,
    load_xor_static_key,
)
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
    """Analyze viewport evidence from a capture session."""
    _test_hooks.setup_rich_logging(level="INFO")

    session_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("capture_session.json")

    try:
        session_text = _load_session(session_path)
    except FileNotFoundError as exc:
        sys.stdout.write(f"{exc}\n")
        raise SystemExit(1) from exc

    session_json = narrow_json_to_dict(load_json_str(session_text))
    session = decode_capture_session(session_json)

    magic = session["magic"]
    if magic is None:
        sys.stdout.write("Capture session has no magic key\n")
        raise SystemExit(1)

    static_key, _ = load_xor_static_key(None)
    if static_key is None:
        sys.stdout.write("Could not load xor_static_key.txt\n")
        raise SystemExit(1)

    xor_table = build_xor_table(static_key, magic)
    result = analyze_capture_session(session, xor_table)
    sys.stdout.write(format_viewport_analysis(result))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()


__all__ = [
    "main",
]
