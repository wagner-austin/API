"""Shared seeded-world builder for the tank-observation tests."""

from __future__ import annotations

from tankpit_bot.state.types import (
    WorldStateDict,
    make_empty_world_state,
    make_tank_state,
)


def make_world_with_seed(
    *,
    tank_id: int = 100,
    x: int = 10,
    y: int = 20,
    team: int = 0,
    rank: int = 0,
    damage_state: int = 0,
    direction: int = 0,
    name: str = "seed",
    is_bot: bool = False,
    timestamp_ms: int = 1000,
    last_wire_seen_ms: int = 900,
    last_position_update_ms: int = 800,
) -> tuple[WorldStateDict, str]:
    """Seed a world state with one tank and return the state + its key.

    Args:
        tank_id: Tank id to seed.
        x: Seeded x coordinate.
        y: Seeded y coordinate.
        team: Seeded team.
        rank: Seeded rank.
        damage_state: Seeded damage tier.
        direction: Seeded direction byte.
        name: Seeded player name.
        is_bot: Seeded bot flag.
        timestamp_ms: Seeded any-source timestamp.
        last_wire_seen_ms: Seeded wire-presence timestamp.
        last_position_update_ms: Seeded position-freshness timestamp.

    Returns:
        ``(WorldStateDict, str)`` where the second element is the
        registry key of the seeded tank.
    """
    state = make_empty_world_state()
    tank = make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        damage_state=damage_state,
        direction=direction,
        name=name,
        is_bot=is_bot,
        is_self=False,
        source="viewport",
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
    )
    state["tanks"][str(tank_id)] = tank
    return state, str(tank_id)
