"""Combat route primitives for the durable HUNT owner.

This module owns typed helper functions for target acquisition, teleport
landing, shoot/miss cycles, and blocked-target replanning. Top-level owner
selection now lives in ``ai_strategy`` and ``hunt_mode``.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_landing import (
    choose_combat_landing_tile,
    has_cardinal_enemy_adjacency,
)
from tankpit_bot.bot.ai.combat_landing import (
    combat_landing_candidates as shared_combat_landing_candidates,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    make_decision,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    EnemyThreatDict,
)
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.sniffer.world_state import is_move_target_failed
from tankpit_bot.state.types import SelfStateDict


def clear_combat_target(ai_state: AIStateDict) -> AIStateDict:
    """Return AI state with combat-target ownership cleared.

    Args:
        ai_state: Current AI state.

    Returns:
        AI state with combat target fields reset.
    """
    return AIStateDict(
        **{
            **ai_state,
            "combat_target_id": -1,
            "combat_target_x": 0,
            "combat_target_y": 0,
        }
    )


def _set_combat_target(
    ai_state: AIStateDict,
    target: EnemyThreatDict,
) -> AIStateDict:
    """Return AI state with a locked combat target.

    Args:
        ai_state: Current AI state.
        target: Combat target to lock.

    Returns:
        AI state with combat target coordinates updated.
    """
    return AIStateDict(
        **{
            **ai_state,
            "combat_target_id": target["tank_id"],
            "combat_target_x": target["x"],
            "combat_target_y": target["y"],
        }
    )


def select_new_combat_target(
    ctx: DecideCtx,
    threats: list[EnemyThreatDict],
) -> EnemyThreatDict | None:
    """Return the next viable new combat target.

    Args:
        ctx: Decision context.
        threats: Visible threats in priority order.

    Returns:
        The next viable enemy target, or ``None`` when combat should not start.
    """
    viable = [
        threat
        for threat in threats
        if str(threat["tank_id"]) not in ctx.blocked_targets
        and str(threat["tank_id"]) not in ctx.killed
    ]
    if not viable:
        return None
    return viable[0]


def get_locked_target(
    ctx: DecideCtx,
    threats: list[EnemyThreatDict],
) -> EnemyThreatDict | None:
    """Find the current combat target in the threat list.

    Args:
        ctx: Decision context.
        threats: Current threat list.

    Returns:
        The locked target if it's still alive and in the world, or None.
    """
    target_id = ctx.ai_state["combat_target_id"]
    if target_id == -1:
        return None
    for t in threats:
        if t["tank_id"] == target_id:
            return t
    return None


def combat_landing_tile(ctx: DecideCtx, target: EnemyThreatDict) -> tuple[int, int]:
    """Choose the tile to teleport to for combat.

    Combat teleports should land adjacent to the enemy rather than on the
    enemy's exact coordinates.

    Args:
        ctx: Decision context.
        target: Enemy threat currently being engaged.

    Returns:
        Tuple of landing coordinates, or (-1, -1) if no landing possible.
    """
    return choose_combat_landing_tile(
        ctx.filtered,
        ctx.self_state,
        target,
        ctx.terrain,
    )


def block_combat_target_and_replan(
    ctx: DecideCtx,
    target: EnemyThreatDict,
) -> TickDecisionDict:
    """Block a combat target and choose the next viable threat.

    Adds the target to blocked_combat_targets so it won't be reacquired until
    the TTL expires. If another viable threat exists, engages that one.
    Otherwise falls back to generic enemy search.

    Args:
        ctx: Decision context.
        target: The unreachable combat target.

    Returns:
        Tick decision for the next viable target, or fallback enemy search.
    """
    blocked = dict(ctx.blocked_targets)
    blocked[str(target["tank_id"])] = ctx.timestamp_ms
    base_with_block = AIStateDict(
        **{
            **clear_combat_target(ctx.base),
            "blocked_combat_targets": blocked,
        }
    )

    threats = analyze_threats(ctx.filtered, ctx.self_state)
    skip = {*blocked, *ctx.killed}
    viable = [t for t in threats if str(t["tank_id"]) not in skip]
    if viable:
        next_target = viable[0]
        emit_ai(
            "blocked %s, switching to %s (id=%d)",
            target["name"],
            next_target["name"],
            next_target["tank_id"],
        )
        return make_decision(
            make_map_open_command(),
            "HUNT",
            800,
            0,
            0,
            f"find {next_target['name']}",
            AIStateDict(
                **{
                    **_set_combat_target(base_with_block, next_target),
                    "last_map_open_ms": ctx.timestamp_ms,
                }
            ),
            ctx.equip,
        )

    emit_ai("blocked %s, no viable threats remaining", target["name"])
    return make_decision(
        make_map_open_command(),
        "HUNT",
        0,
        0,
        0,
        "find_enemies",
        AIStateDict(**{**base_with_block, "last_map_open_ms": ctx.timestamp_ms}),
        ctx.equip,
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _combat_open_map(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase 0: Open map to get fresh enemy positions."""
    emit_ai("open map to find %s", target["name"])
    return make_decision(
        make_map_open_command(),
        "HUNT",
        800,
        0,
        0,
        f"find {target['name']}",
        AIStateDict(
            **{
                **_set_combat_target(ctx.base, target),
                "last_map_open_ms": ctx.timestamp_ms,
            }
        ),
        ctx.equip,
    )


def open_map_for_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Open the map to refresh or acquire the given target.

    Args:
        ctx: Decision context.
        target: Combat target to refresh.

    Returns:
        Map-open decision that locks the target.
    """
    return _combat_open_map(ctx, target)


def _combat_teleport(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict | None:
    """Phase 1: Teleport to enemy."""
    landing_x, landing_y = combat_landing_tile(ctx, target)
    if landing_x == -1 and landing_y == -1:
        emit_ai("no combat landing tile for %s, blocking target", target["name"])
        return block_combat_target_and_replan(ctx, target)
    if is_move_target_failed(landing_x, landing_y, ctx.timestamp_ms):
        emit_ai(
            "combat landing (%d,%d) for %s already failed, blocking target",
            landing_x,
            landing_y,
            target["name"],
        )
        return block_combat_target_and_replan(ctx, target)
    if not can_afford_teleport(
        ctx,
        landing_x,
        landing_y,
        reserve_fuel=ctx.config["hunt_min_fuel"],
    ):
        emit_ai(
            "cannot afford combat teleport for %s to (%d,%d) (fuel=%d cost=%d reserve=%d)",
            target["name"],
            landing_x,
            landing_y,
            ctx.fuel,
            teleport_fuel_cost_to(ctx, landing_x, landing_y),
            ctx.config["hunt_min_fuel"],
        )
        return None
    emit_ai("teleport near %s to (%d,%d)", target["name"], landing_x, landing_y)
    return make_decision(
        make_teleport_command(landing_x, landing_y),
        "HUNT",
        800,
        landing_x,
        landing_y,
        f"teleport {target['name']}",
        _set_combat_target(ctx.base, target),
        ctx.equip,
    )


def teleport_to_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict | None:
    """Teleport toward the given combat target when legal.

    Args:
        ctx: Decision context.
        target: Combat target to close on.

    Returns:
        Teleport decision, a blocked-target replanning decision, or ``None``
        when the teleport is unaffordable.
    """
    return _combat_teleport(ctx, target)


def _combat_close(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict | None:
    """Phase closing: confirm geometry before shooting."""
    if has_cardinal_combat_shot(ctx.self_state, target):
        return _combat_shoot(ctx, target)
    emit_ai(
        "not in cardinal firing position for %s from (%d,%d); re-closing",
        target["name"],
        ctx.self_state["x"],
        ctx.self_state["y"],
    )
    return _combat_teleport(ctx, target)


def close_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict | None:
    """Close distance on the given combat target.

    Args:
        ctx: Decision context.
        target: Combat target to approach.

    Returns:
        Close-range combat decision, or ``None`` when no close action is legal.
    """
    return _combat_close(ctx, target)


def _combat_shoot(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase engaging: Shoot. On miss: reacquire."""
    if ctx.combat_feedback == "miss":
        emit_ai("miss - reopening map for %s", target["name"])
        return _combat_open_map(ctx, target)

    emit_ai("shoot %s at (%d,%d)", target["name"], target["x"], target["y"])
    engaging_state = _set_combat_target(ctx.base, target)
    return make_decision(
        make_shoot_command(target["x"], target["y"], target["tank_id"]),
        "HUNT",
        800,
        target["x"],
        target["y"],
        f"shoot {target['name']}",
        AIStateDict(
            **{
                **engaging_state,
                "last_shoot_ms": ctx.timestamp_ms,
                "last_shot_target_id": target["tank_id"],
                "last_shot_target_name": target["name"],
            }
        ),
        ctx.equip,
    )


def engage_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Engage the given combat target.

    Args:
        ctx: Decision context.
        target: Combat target to shoot at.

    Returns:
        Combat engage decision, including miss-driven refresh behavior.
    """
    return _combat_shoot(ctx, target)


def _combat_landing_candidates(
    ctx: DecideCtx,
    target: EnemyThreatDict,
) -> list[tuple[int, int]]:
    """Return usable adjacent landing tiles ordered by distance to self."""
    return shared_combat_landing_candidates(ctx.filtered, ctx.self_state, target)


def has_cardinal_combat_shot(
    self_state: SelfStateDict,
    target: EnemyThreatDict,
) -> bool:
    """Return True when self is cardinally adjacent to the target.

    Args:
        self_state: Player's own state.
        target: Enemy threat.

    Returns:
        True if Manhattan distance is exactly 1.
    """
    return has_cardinal_enemy_adjacency(self_state, target)


__all__ = [
    "block_combat_target_and_replan",
    "clear_combat_target",
    "close_target",
    "combat_landing_tile",
    "engage_target",
    "get_locked_target",
    "has_cardinal_combat_shot",
    "open_map_for_target",
    "select_new_combat_target",
    "teleport_to_target",
]
