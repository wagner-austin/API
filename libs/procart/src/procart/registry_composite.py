from __future__ import annotations

from typing import Final

_NAMES: Final[tuple[str, ...]] = (
    "normal",
    "add",
    "screen",
    "lighten",
    "darken",
)


def list_available_composite_ops() -> list[str]:
    """List supported composite operation names.

    Returns:
        list[str]: Stable list of composite identifiers.
    """

    return list(_NAMES)


__all__ = [
    "list_available_composite_ops",
]
