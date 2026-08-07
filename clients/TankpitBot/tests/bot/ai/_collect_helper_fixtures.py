"""Shared enemy builder for the COLLECT helper tests."""

from __future__ import annotations

from tankpit_bot.state.types import TankStateDict


def _enemy(*, x: int, y: int, timestamp_ms: int = 100000) -> TankStateDict:
    """Create a visible enemy tank for helper tests.

    Args:
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        timestamp_ms: Observation timestamp.

    Returns:
        Enemy tank state.
    """
    from tankpit_bot.state.types import make_tank_state

    return make_tank_state(
        tank_id=50,
        x=x,
        y=y,
        team=2,
        rank=1,
        name="Enemy",
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=timestamp_ms,
        last_position_update_ms=timestamp_ms,
        last_viewport_observation_ms=timestamp_ms,
    )
