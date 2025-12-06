"""Strictly typed JSON utilities that avoid Any contamination.

Follows qr-api pattern for handling json.loads() with recursive TypeAlias.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue, load_json_str

__all__ = ["JSONValue", "parse_json_dict"]


def parse_json_dict(s: str) -> dict[str, JSONValue] | None:
    """Parse JSON string, returning dict or None if not a dict."""
    value: JSONValue = load_json_str(s)
    return value if isinstance(value, dict) else None
