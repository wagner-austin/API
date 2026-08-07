"""Shared enemy builder for the COLLECT integration tests."""

from __future__ import annotations

from tankpit_bot.state.types import (
    TankStateDict,
    make_tank_state,
)


def _enemy(
    *,
    tank_id: int = 50,
    x: int = 103,
    y: int = 103,
    name: str = "red-1",
    timestamp_ms: int = 100000,
) -> TankStateDict:
    """Create a visible enemy tank for recovery arbitration tests.

    Args:
        tank_id: Enemy tank id.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.
        timestamp_ms: Observation timestamp in milliseconds.

    Returns:
        Enemy tank state.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=1,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=timestamp_ms,
        last_position_update_ms=timestamp_ms,
        last_viewport_observation_ms=timestamp_ms,
    )
