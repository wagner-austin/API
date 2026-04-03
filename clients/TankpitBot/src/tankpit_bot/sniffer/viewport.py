"""Viewport origin tracking for sniffer-side absolute coordinate helpers.

The authoritative viewport origin comes from protocol ``0x5A``. Sniffer
formatting and registry helpers use this module-level state to convert
viewport-relative coordinates into absolute world coordinates.
"""

from __future__ import annotations

_viewport_left: int | None = None
_viewport_top: int | None = None


def update_viewport_origin(viewport_left: int, viewport_top: int) -> None:
    """Store the authoritative viewport origin from protocol ``0x5A``.

    Args:
        viewport_left: Absolute left edge of the observable viewport.
        viewport_top: Absolute top edge of the observable viewport.
    """
    global _viewport_left, _viewport_top
    _viewport_left = viewport_left
    _viewport_top = viewport_top


def get_viewport_left() -> int | None:
    """Get current viewport left edge x coordinate.

    Returns:
        Viewport left edge x coordinate, or None if not yet determined.
    """
    return _viewport_left


def get_viewport_top() -> int | None:
    """Get current viewport top edge y coordinate.

    Returns:
        Viewport top edge y coordinate, or None if not yet determined.
    """
    return _viewport_top


def reset_viewport_tracking() -> None:
    """Reset viewport tracking state.

    Called when starting a new session or leaving the game.
    """
    global _viewport_left, _viewport_top
    _viewport_left = None
    _viewport_top = None


__all__ = [
    "get_viewport_left",
    "get_viewport_top",
    "reset_viewport_tracking",
    "update_viewport_origin",
]
