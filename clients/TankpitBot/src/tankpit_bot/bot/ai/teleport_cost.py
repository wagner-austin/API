"""Teleport fuel cost helpers for AI planning."""

from __future__ import annotations

from math import isqrt


def compute_teleport_fuel_cost(
    start_x: int,
    start_y: int,
    target_x: int,
    target_y: int,
) -> int:
    """Compute the exact fuel cost for a teleport.

    Tankpit teleport fuel cost scales with Euclidean distance:

    ``floor(6 * sqrt(dx^2 + dy^2))``

    This implementation uses integer square root over ``36 * distance_sq`` so
    the returned value is exact without floating-point drift.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        target_x: Destination X coordinate.
        target_y: Destination Y coordinate.

    Returns:
        Exact integer fuel cost for the teleport.
    """
    delta_x = target_x - start_x
    delta_y = target_y - start_y
    distance_sq = delta_x * delta_x + delta_y * delta_y
    return isqrt(36 * distance_sq)


__all__ = ["compute_teleport_fuel_cost"]
