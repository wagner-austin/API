"""Coordinate-key string helpers shared across world-state collections.

World-state dicts are keyed by ``"x,y"`` strings so JSON serialization
stays straightforward. These helpers centralize the format so adopters
never hand-roll the join/split logic.
"""

from __future__ import annotations


def coord_key(x: int, y: int) -> str:
    """Create a coordinate key string for dict indexing.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        String key in format "x,y".
    """
    return f"{x},{y}"


def parse_coord_key(key: str) -> tuple[int, int]:
    """Parse a coordinate key string.

    Args:
        key: String key in format "x,y".

    Returns:
        Tuple of (x, y) coordinates.

    Raises:
        ValueError: If key format is invalid.
    """
    parts = key.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coord key format: {key}")
    return int(parts[0]), int(parts[1])


def viewport_scan_key(left: int, top: int) -> str:
    """Create a viewport-origin key string for scan coverage indexing.

    Args:
        left: Viewport left X coordinate.
        top: Viewport top Y coordinate.

    Returns:
        String key in format "left,top".
    """
    return f"{left},{top}"


__all__ = [
    "coord_key",
    "parse_coord_key",
    "viewport_scan_key",
]
