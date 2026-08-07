"""Shared world/self builder for the equipment search tests."""

from __future__ import annotations

from tankpit_bot.state.types import (
    SelfStateDict,
    WorldStateDict,
    make_self_state,
    make_viewport_state,
)


def _world_and_self(x: int = 100, y: int = 100) -> tuple[WorldStateDict, SelfStateDict]:
    """Create empty world and self state at given position.

    Args:
        x: Self X coordinate.
        y: Self Y coordinate.

    Returns:
        Tuple of (empty WorldStateDict, SelfStateDict).
    """
    vp_left = x - 9
    vp_top = y - 9
    world = WorldStateDict(
        self_state=None,
        tanks={},
        containers={},
        mines={},
        terrain={},
        viewport=make_viewport_state(left=vp_left, top=vp_top, width=18, height=18),
        scanned_tiles={
            f"{tx},{ty}": 100000
            for ty in range(vp_top, vp_top + 18)
            for tx in range(vp_left, vp_left + 18)
        },
        timestamp_ms=0,
    )
    state = make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=0,
        rank=4,
        fuel=800,
        leaderboard_position=1,
    )
    return world, state
