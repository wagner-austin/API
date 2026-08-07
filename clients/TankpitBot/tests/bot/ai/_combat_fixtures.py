"""Shared enemy-threat builder for the combat test modules.

``test_combat_strategy.py`` was 1,719 lines; it is now four modules
over this one builder.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.world_types import EnemyThreatDict


def _enemy_threat(
    *,
    tank_id: int = 50,
    x: int = 120,
    y: int = 100,
    name: str = "Enemy",
    last_wire_seen_ms: int = 100000,
    last_position_update_ms: int = 100000,
) -> EnemyThreatDict:
    """Create a typed enemy threat for combat helper tests.

    Args:
        tank_id: Enemy tank identifier.
        x: Enemy x coordinate.
        y: Enemy y coordinate.
        name: Enemy display name.
        last_wire_seen_ms: Last wire-presence confirmation (defaults to
            the helper's ``timestamp_ms`` so the threat is wire-present at
            the tests' 100000 clock unless overridden to model a ghost).
        last_position_update_ms: Last wire-sourced position confirmation
            (defaults to the helper's clock so the threat passes the
            kill-shot gate unless overridden to model a stale-position
            target).

    Returns:
        Enemy threat payload.
    """
    return EnemyThreatDict(
        tank_id=tank_id,
        x=x,
        y=y,
        distance=abs(x - 100) + abs(y - 100),
        damage_state=0,
        rank=1,
        team=2,
        name=name,
        is_bot=False,
        timestamp_ms=100000,
        last_wire_seen_ms=last_wire_seen_ms,
        last_position_update_ms=last_position_update_ms,
        last_aim_x=-1,
        last_aim_y=-1,
        last_aim_weapon=-1,
        last_aim_ms=0,
    )
