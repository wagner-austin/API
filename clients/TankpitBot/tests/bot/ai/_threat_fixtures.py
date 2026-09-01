"""Shared threat and world builders for the threat tests."""

from __future__ import annotations

from tankpit_bot.state.types import (
    SelfStateDict,
    TankStateDict,
    WorldStateDict,
    make_self_state,
    make_tank_state,
    make_viewport_state,
)
from tankpit_bot.types.constants import TankLiveness


def _tank(
    key: str,
    x: int = 0,
    y: int = 0,
    team: int = 1,
    damage_state: int = 0,
    direction: int = 0,
    name: str = "",
    is_bot: bool = True,
    is_self: bool = False,
    liveness: TankLiveness = "alive",
    rank: int = 1,
) -> TankStateDict:
    """Create a TankStateDict with defaults for testing.

    Args:
        key: Tank ID as string.
        x: X coordinate.
        y: Y coordinate.
        team: Team ID.
        damage_state: Damage state (0-3).
        direction: Sprite direction (0-31 alive, 32-33 dead).
        name: Player name (defaults to "tank-{key}").
        is_bot: Whether this is a bot.
        is_self: Whether this is the player's tank.
        liveness: Lifecycle state. Defaults to ``"alive"``.
        rank: Military rank (0 recruit .. 8). Defaults to 1 — a
            points-paying tier, so ordering tests that are not about
            the recruit rule stay on pure distance.

    Returns:
        TankStateDict with the provided values.
    """
    # Default names are practice-bot style (``red-{key}``): these
    # fixtures test the distance/freshness/liveness gates, and bot
    # classification keeps them clear of BOTH human-only gates (the
    # 2026-07-28 rank window and the 2026-07-30 consent contract).
    # Tests about the human gates pass explicit human names.
    return make_tank_state(
        tank_id=int(key),
        x=x,
        y=y,
        team=team,
        rank=rank,
        damage_state=damage_state,
        direction=direction,
        name=name or f"red-{key}",
        is_bot=is_bot,
        is_self=is_self,
        liveness=liveness,
    )


def _world(tanks: dict[str, TankStateDict]) -> WorldStateDict:
    """Build a WorldStateDict with only the given tanks.

    Args:
        tanks: Dict mapping tank_id string keys to TankStateDicts.

    Returns:
        WorldStateDict with the provided tanks.
    """
    return WorldStateDict(
        self_state=None,
        tanks=tanks,
        containers={},
        mines={},
        terrain={},
        viewport=make_viewport_state(left=0, top=0, width=18, height=18),
        scanned_tiles={},
        timestamp_ms=0,
    )


def _self_at(x: int = 100, y: int = 100) -> SelfStateDict:
    """Create self state at given position on team 0."""
    return make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=0,
        rank=4,
        fuel=800,
        leaderboard_position=1,
    )
