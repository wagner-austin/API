"""AI views of the world: enemy threats and path steps.

The two records the AI derives from world state and passes between
its stages. Kept apart from the session-scoped
:mod:`tankpit_bot.bot.ai.types` because these describe what the bot
SEES, not what it remembers.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class EnemyThreatDict(TypedDict):
    """An analyzed enemy tank with computed distance.

    Carries the three freshness timestamps from
    :class:`tankpit_bot.state.types.tank.TankStateDict` so the
    combat-strategy layer can read them without re-querying the
    registry.

    Attributes:
        tank_id: Enemy tank id.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.
        distance: Manhattan distance from self.
        damage_state: Fuel-quartile health tier (0=near death .. 3=full).
        rank: Military rank (0 recruit .. 8 general). Lower rank = weaker.
        team: Enemy team id (0-3).
        name: Enemy player name.
        is_bot: Whether this enemy is a bot.
        timestamp_ms: When this tank was last confirmed by ANY source.
            Drives acquisition freshness.
        last_wire_seen_ms: When a wire-presence source last vouched the
            tank is in view. Drives the ghost gate.
        last_position_update_ms: When a wire-sourced observation last
            carried fresh ``(x, y)``. Drives the kill-shot gate.
        last_aim_x: Wire-reported barrel-aim X from this enemy's most
            recent 0x53 ShootEvent, or ``-1`` when never seen firing.
            Combat consumers (avoid-fire, predicted-LOS) read this to
            reason about which tile the enemy may target next.
        last_aim_y: Wire-reported barrel-aim Y from the same event.
        last_aim_weapon: Weapon byte from the same event
            (0=single, 1=dual, 2=missile, 3=homing). ``-1`` when never
            seen firing.
        last_aim_ms: Wall-clock of the most recent 0x53 ShootEvent
            attributed to this enemy. Consumers should age the aim
            with their own staleness threshold.
    """

    tank_id: int
    x: int
    y: int
    distance: int
    damage_state: int
    rank: int
    team: int
    name: str
    is_bot: bool
    timestamp_ms: int
    last_wire_seen_ms: int
    last_position_update_ms: int
    last_aim_x: int
    last_aim_y: int
    last_aim_weapon: int
    last_aim_ms: int


def make_enemy_threat(
    tank_id: int,
    x: int,
    y: int,
    distance: int,
    damage_state: int,
    rank: int,
    team: int,
    name: str,
    is_bot: bool,
    timestamp_ms: int = 0,
    last_wire_seen_ms: int = 0,
    last_position_update_ms: int = 0,
    last_aim_x: int = -1,
    last_aim_y: int = -1,
    last_aim_weapon: int = -1,
    last_aim_ms: int = 0,
) -> EnemyThreatDict:
    """Create an EnemyThreatDict.

    Args:
        tank_id: Enemy tank id.
        x: Enemy X coordinate.
        y: Enemy Y coordinate.
        distance: Manhattan distance from self.
        damage_state: Fuel-quartile health tier (0=near death .. 3=full).
        rank: Military rank (0 recruit .. 8 general).
        team: Team id (0-3).
        name: Player name.
        is_bot: Whether this is a bot.
        timestamp_ms: When this tank was last confirmed by any source.
        last_wire_seen_ms: When a wire-presence source last vouched the
            tank is in view. Zero means never wire-confirmed.
        last_position_update_ms: When a wire-sourced observation last
            carried fresh ``(x, y)``. Zero means position has never
            been wire-confirmed.
        last_aim_x: Wire-reported barrel-aim X from the enemy's most
            recent 0x53 ShootEvent. Defaults to ``-1`` (never seen).
        last_aim_y: Wire-reported barrel-aim Y from the same event.
        last_aim_weapon: Weapon byte (0=single, 1=dual, 2=missile,
            3=homing) from the same event. ``-1`` when never seen.
        last_aim_ms: Wall-clock of the most recent 0x53 event for
            this enemy. ``0`` when never seen.

    Returns:
        EnemyThreatDict with the provided values.
    """
    return EnemyThreatDict(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=distance,
        damage_state=damage_state,
        rank=rank,
        team=team,
        name=name,
        is_bot=is_bot,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
        last_aim_x=last_aim_x,
        last_aim_y=last_aim_y,
        last_aim_weapon=last_aim_weapon,
        last_aim_ms=last_aim_ms,
    )


class PathStepDict(TypedDict):
    """A single step in a computed path.

    Attributes:
        x: X coordinate of this step.
        y: Y coordinate of this step.
    """

    x: int
    y: int


def make_path_step(x: int, y: int) -> PathStepDict:
    """Create a PathStepDict.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        PathStepDict with the provided values.
    """
    return PathStepDict(x=x, y=y)


__all__ = [
    "EnemyThreatDict",
    "PathStepDict",
    "make_enemy_threat",
    "make_path_step",
]
