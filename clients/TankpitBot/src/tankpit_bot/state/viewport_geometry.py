"""Helpers for visible viewport and radar envelope geometry."""

from __future__ import annotations

from tankpit_bot.state.types import ViewportStateDict

VISIBLE_VIEWPORT_WIDTH = 16
VISIBLE_VIEWPORT_HEIGHT = 16
REGULAR_RADAR_RADIUS = 3
RADAR_ENVELOPE_MARGIN = 1
VIEWPORT_PATCH_WIDTH = VISIBLE_VIEWPORT_WIDTH + (RADAR_ENVELOPE_MARGIN * 2)
VIEWPORT_PATCH_HEIGHT = VISIBLE_VIEWPORT_HEIGHT + (RADAR_ENVELOPE_MARGIN * 2)


def make_visible_viewport_state(left: int, top: int) -> ViewportStateDict:
    """Build the canonical visible viewport state.

    Args:
        left: Visible viewport left edge X coordinate.
        top: Visible viewport top edge Y coordinate.

    Returns:
        ViewportStateDict for the visible 16x16 viewport.
    """
    return ViewportStateDict(
        left=left,
        top=top,
        width=VISIBLE_VIEWPORT_WIDTH,
        height=VISIBLE_VIEWPORT_HEIGHT,
    )


def viewport_visible_bounds(viewport: ViewportStateDict) -> tuple[int, int, int, int]:
    """Return inclusive visible viewport bounds.

    Args:
        viewport: Current visible viewport state.

    Returns:
        Inclusive ``(left, top, right, bottom)`` visible bounds.
    """
    left = viewport["left"]
    top = viewport["top"]
    right = left + viewport["width"] - 1
    bottom = top + viewport["height"] - 1
    return (left, top, right, bottom)


def viewport_radar_bounds(viewport: ViewportStateDict) -> tuple[int, int, int, int]:
    """Return inclusive radar coverage bounds.

    Radar extends one tile beyond the visible viewport in every direction.

    Args:
        viewport: Current visible viewport state.

    Returns:
        Inclusive ``(left, top, right, bottom)`` radar coverage bounds.
    """
    left, top, right, bottom = viewport_visible_bounds(viewport)
    return (
        left - RADAR_ENVELOPE_MARGIN,
        top - RADAR_ENVELOPE_MARGIN,
        right + RADAR_ENVELOPE_MARGIN,
        bottom + RADAR_ENVELOPE_MARGIN,
    )


def regular_radar_bounds(center_x: int, center_y: int) -> tuple[int, int, int, int]:
    """Return inclusive bounds for the built-in 7x7 radar scan.

    Args:
        center_x: Controlled tank X coordinate.
        center_y: Controlled tank Y coordinate.

    Returns:
        Inclusive ``(left, top, right, bottom)`` local radar bounds.
    """
    return (
        center_x - REGULAR_RADAR_RADIUS,
        center_y - REGULAR_RADAR_RADIUS,
        center_x + REGULAR_RADAR_RADIUS,
        center_y + REGULAR_RADAR_RADIUS,
    )


def viewport_patch_world_coords(
    viewport_left: int,
    viewport_top: int,
    col: int,
    row: int,
) -> tuple[int, int]:
    """Translate decoded ``0x5A`` patch coordinates into world coordinates.

    The protocol stream itself uses an 18-wide delta grid, but the live bot's
    world-state application currently treats the decoded ``col`` and ``row``
    values as direct offsets from the packet viewport origin.

    Args:
        viewport_left: Absolute visible viewport left edge.
        viewport_top: Absolute visible viewport top edge.
        col: Patch-grid column from the decoded ``0x5A`` entity.
        row: Patch-grid row from the decoded ``0x5A`` entity.

    Returns:
        Absolute world ``(x, y)`` for the patch cell.
    """
    return (
        viewport_left + col - RADAR_ENVELOPE_MARGIN,
        viewport_top + row - RADAR_ENVELOPE_MARGIN,
    )


__all__ = [
    "RADAR_ENVELOPE_MARGIN",
    "REGULAR_RADAR_RADIUS",
    "VIEWPORT_PATCH_HEIGHT",
    "VIEWPORT_PATCH_WIDTH",
    "VISIBLE_VIEWPORT_HEIGHT",
    "VISIBLE_VIEWPORT_WIDTH",
    "make_visible_viewport_state",
    "regular_radar_bounds",
    "viewport_patch_world_coords",
    "viewport_radar_bounds",
    "viewport_visible_bounds",
]
