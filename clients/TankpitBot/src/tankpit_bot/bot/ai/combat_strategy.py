"""Combat decision logic for AI strategy.

Handles target acquisition, teleport landing, shoot/miss cycles, and
blocked-target replanning. All functions operate on a ``DecideCtx``.
"""

from __future__ import annotations

from collections.abc import Callable

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    equipment_reserve_restored,
    has_recent_map_snapshot,
    make_decision,
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


def try_combat(ctx: DecideCtx) -> TickDecisionDict | None:
    """Route to the correct combat phase.

    Args:
        ctx: Decision context.

    Returns:
        Tick decision for a combat action, or None if combat is not viable.
    """
    if ctx.fuel < ctx.config["fuel_low_threshold"] and not combat_in_progress(ctx):
        return None

    threats = analyze_threats(ctx.filtered, ctx.self_state)
    if not threats:
        return None

    target = get_locked_target(ctx, threats)
    if target is None:
        if not equipment_reserve_restored(ctx):
            return None
        viable = [
            t
            for t in threats
            if str(t["tank_id"]) not in ctx.blocked_targets and str(t["tank_id"]) not in ctx.killed
        ]
        if not viable:
            return None
        target = viable[0]
        emit_ai("new target %s (id=%d)", target["name"], target["tank_id"])
        if has_recent_map_snapshot(ctx):
            emit_ai("fresh map intel available - teleporting to %s", target["name"])
            return _combat_teleport(ctx, target)
        return _combat_open_map(ctx, target)

    phase = ctx.ai_state["combat_phase"]

    if phase == "engaging":
        return _combat_shoot(ctx, target)
    if phase == "closing":
        return _combat_close(ctx, target)
    return _combat_open_map(ctx, target)


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


def combat_in_progress(ctx: DecideCtx) -> bool:
    """Return True when the AI is in an active combat engagement.

    Returns False when combat is no longer executable:
    - dual depleted (can't shoot)
    - locked target was killed (should recover before reacquiring)

    This releases the combat lock so the planner can fall through to
    equipment/fuel recovery instead of immediately reacquiring.

    Args:
        ctx: Decision context.

    Returns:
        True if combat is actively locked and executable.
    """
    target_id = ctx.ai_state["combat_target_id"]
    return (
        target_id != -1
        and str(target_id) not in ctx.killed
        and ctx.ai_state["combat_phase"] in ("closing", "engaging")
        and ctx.inventory["dual_shots"]["count"] != 0
    )


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
    candidates = _combat_landing_candidates(ctx, target)
    if not candidates:
        return (-1, -1)

    if ctx.terrain is not None:
        for candidate_x, candidate_y in candidates:
            if ctx.terrain.is_passable(candidate_x, candidate_y):
                return (candidate_x, candidate_y)
        return (-1, -1)

    return candidates[0]


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
            **ctx.base,
            "blocked_combat_targets": blocked,
            "combat_target_id": -1,
            "combat_target_x": 0,
            "combat_target_y": 0,
            "combat_phase": "none",
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
                    **base_with_block,
                    "combat_target_id": next_target["tank_id"],
                    "combat_target_x": next_target["x"],
                    "combat_target_y": next_target["y"],
                    "last_map_open_ms": ctx.timestamp_ms,
                    "combat_phase": "closing",
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
                **ctx.base,
                "combat_target_id": target["tank_id"],
                "combat_target_x": target["x"],
                "combat_target_y": target["y"],
                "last_map_open_ms": ctx.timestamp_ms,
                "combat_phase": "closing",
            }
        ),
        ctx.equip,
    )


def _combat_teleport(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
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
    emit_ai("teleport near %s to (%d,%d)", target["name"], landing_x, landing_y)
    return make_decision(
        make_teleport_command(landing_x, landing_y),
        "HUNT",
        800,
        landing_x,
        landing_y,
        f"teleport {target['name']}",
        AIStateDict(
            **{
                **ctx.base,
                "combat_target_id": target["tank_id"],
                "combat_target_x": target["x"],
                "combat_target_y": target["y"],
                "combat_phase": "closing",
            }
        ),
        ctx.equip,
    )


def _combat_close(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
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


def _combat_shoot(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase engaging: Shoot. On miss: reacquire."""
    if ctx.combat_feedback == "miss":
        emit_ai("miss - reopening map for %s", target["name"])
        return _combat_open_map(ctx, target)

    emit_ai("shoot %s at (%d,%d)", target["name"], target["x"], target["y"])
    return make_decision(
        make_shoot_command(target["x"], target["y"], target["tank_id"]),
        "HUNT",
        800,
        target["x"],
        target["y"],
        f"shoot {target['name']}",
        AIStateDict(
            **{
                **ctx.base,
                "combat_target_id": target["tank_id"],
                "combat_target_x": target["x"],
                "combat_target_y": target["y"],
                "last_shoot_ms": ctx.timestamp_ms,
                "last_shot_target_id": target["tank_id"],
                "last_shot_target_name": target["name"],
                "combat_phase": "engaging",
            }
        ),
        ctx.equip,
    )


def _combat_landing_candidates(
    ctx: DecideCtx,
    target: EnemyThreatDict,
) -> list[tuple[int, int]]:
    """Return usable adjacent landing tiles ordered by distance to self."""
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    candidates = [
        (target["x"] + 1, target["y"]),
        (target["x"] - 1, target["y"]),
        (target["x"], target["y"] + 1),
        (target["x"], target["y"] - 1),
    ]
    usable: list[tuple[int, int]] = []
    for candidate_x, candidate_y in candidates:
        if not (0 <= candidate_x <= 255 and 0 <= candidate_y <= 255):
            continue
        if _is_dynamically_occupied(ctx, candidate_x, candidate_y):
            continue
        usable.append((candidate_x, candidate_y))
    usable.sort(key=_combat_distance_key(sx, sy))
    return usable


def _combat_distance_key(sx: int, sy: int) -> Callable[[tuple[int, int]], int]:
    """Return a stable Manhattan-distance key for combat landing sort."""

    def key(pos: tuple[int, int]) -> int:
        return abs(pos[0] - sx) + abs(pos[1] - sy)

    return key


def _is_dynamically_occupied(ctx: DecideCtx, x: int, y: int) -> bool:
    """Return True when a tile is occupied by a tank, container, or mine."""
    if any(tank["x"] == x and tank["y"] == y for tank in ctx.filtered["tanks"].values()):
        return True
    if f"{x},{y}" in ctx.world["containers"]:
        return True
    return f"{x},{y}" in ctx.world["mines"]


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
    return abs(self_state["x"] - target["x"]) + abs(self_state["y"] - target["y"]) == 1


__all__ = [
    "block_combat_target_and_replan",
    "combat_in_progress",
    "combat_landing_tile",
    "get_locked_target",
    "has_cardinal_combat_shot",
    "try_combat",
]
