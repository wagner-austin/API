"""Helpers for visible viewport and radar envelope geometry."""

from __future__ import annotations

from tankpit_bot.state.rank_formulas import free_radar_radius
from tankpit_bot.state.types import ViewportStateDict, make_viewport_state

VISIBLE_VIEWPORT_WIDTH = 16
VISIBLE_VIEWPORT_HEIGHT = 16
RADAR_ENVELOPE_MARGIN = 1
VIEWPORT_PATCH_WIDTH = VISIBLE_VIEWPORT_WIDTH + (RADAR_ENVELOPE_MARGIN * 2)
VIEWPORT_PATCH_HEIGHT = VISIBLE_VIEWPORT_HEIGHT + (RADAR_ENVELOPE_MARGIN * 2)


def make_visible_viewport_state(left: int, top: int, observed_ms: int = 0) -> ViewportStateDict:
    """Build the canonical visible viewport state.

    Args:
        left: Visible viewport left edge X coordinate.
        top: Visible viewport top edge Y coordinate.
        observed_ms: When the 0x5A update that set this viewport
            arrived. Zero for fixtures constructed without a clock.

    Returns:
        ViewportStateDict for the visible 16x16 viewport.
    """
    return make_viewport_state(
        left=left,
        top=top,
        width=VISIBLE_VIEWPORT_WIDTH,
        height=VISIBLE_VIEWPORT_HEIGHT,
        observed_ms=observed_ms,
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


def regular_radar_bounds(
    center_x: int,
    center_y: int,
    rank: int,
) -> tuple[int, int, int, int]:
    """Return inclusive bounds for the built-in radar scan.

    The built-in radar radius is rank-scaled: chebyshev
    ``2 + rank // 3`` (5x5 / 7x7 / 9x9 at rank bands 0-2 / 3-5 / 6-8).
    Only the extra radar sweeps the whole viewport regardless of rank.
    See :func:`tankpit_bot.state.rank_formulas.free_radar_radius` for
    the mining chain.

    Args:
        center_x: Controlled tank X coordinate.
        center_y: Controlled tank Y coordinate.
        rank: Controlled tank rank (``self_state["rank"]``, 0..8).

    Returns:
        Inclusive ``(left, top, right, bottom)`` local radar bounds.
    """
    radius = free_radar_radius(rank)
    return (
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
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
