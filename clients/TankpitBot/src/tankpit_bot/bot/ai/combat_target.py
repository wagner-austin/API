"""Combat target selection, locking, and blocking for the HUNT owner.

The "who do we fight" layer: shot-line and engagement predicates, the
locked-target accessors, landing-tile choice, and the blocklist
replan. Depends on no other combat module, so it is the base every
other combat stage builds on.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.combat_landing import (
    SHOT_RANGE_TILES,
    choose_combat_landing_tile,
)
from tankpit_bot.bot.ai.combat_landing import (
    combat_landing_candidates as shared_combat_landing_candidates,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    make_decision,
)
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    make_map_open_command,
)
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state.line_of_sight import is_shot_line_clear


def has_clear_shot_line(ctx: DecideCtx, target: EnemyThreatDict) -> bool:
    """Return True when the straight line to the target has no occluder.

    The user's firing law has always carried a clearance clause ("as
    long as theyre on the viewport and its a CLEAR dual shot") --
    enforced since flag s3-16 through the lifted
    :func:`tankpit_bot.state.line_of_sight.is_shot_line_clear` test:
    rock and movable land blocks occlude, water and mines never do,
    and cardinal adjacency is trivially clear (no intermediate
    tiles). An in-view target behind terrain is NOT shot from here --
    spending an over-terrain homing where a re-close buys point-blank
    duals is the F11 weapon-economy hole ([[flag-triage-20260729]]).

    Args:
        ctx: Decision context.
        target: Enemy threat under aim.

    Returns:
        True when a dual fired from the current tile flies clean.
    """
    return is_shot_line_clear(
        ctx.self_state["x"],
        ctx.self_state["y"],
        target["x"],
        target["y"],
        ctx.terrain,
        ctx.world["terrain"],
    )


def is_already_engaged(ctx: DecideCtx) -> bool:
    """Return True when the bot has already dispatched a shot at the locked target.

    The discriminator is ``last_shot_target_id`` -- set only when
    :func:`engage_target` actually dispatches a ``shoot`` command for
    the current ``combat_target_id``. A match proves the bot is in a
    mid-fight stay-put scenario rather than a fresh acquisition: the
    initial teleport has happened, at least one shot has resolved, and
    the target is now somewhere other than point-blank. In that case
    the right move is to keep firing (the server picks ``homing`` when
    not adjacent, and homing tracks) instead of teleporting to chase
    a moving enemy.

    A mismatch means the lock is pre-engagement -- either a fresh
    acquisition that has not yet teleported, or a re-acquire after
    a kill -- and the planner should produce the initial close
    teleport rather than fire from afar.

    Args:
        ctx: Decision context.

    Returns:
        True if ``last_shot_target_id`` equals ``combat_target_id``.
    """
    return ctx.ai_state["last_shot_target_id"] == ctx.ai_state["combat_target_id"]


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


def has_standoff_landing(
    x: int,
    y: int,
    terrain: TerrainMapProtocol | None,
) -> bool:
    """Return True when a passable landing exists within shot range.

    The engageability question is stand-off, not adjacency: the close
    teleports onto the target's own tile and the server displaces the
    bot to the nearest open ground, and duals fire from any in-view
    tile within ``SHOT_RANGE_TILES`` ([[weapon-selection]]). A target
    is therefore viable as long as SOME passable tile lies within the
    shot-range diamond. The stricter passable-adjacent form of this
    gate made mine-ringed players invisible to acquisition (live
    2026-07-29: Yuppler ringed himself with mines and every pass
    rejected him with no_passable_adjacent -- the mine-composed
    passability view marks the whole ring impassable -- so the human
    preempt never even saw him while the bot farmed practice bots).
    """
    if terrain is None:
        return True
    for dx in range(-SHOT_RANGE_TILES, SHOT_RANGE_TILES + 1):
        remaining = SHOT_RANGE_TILES - abs(dx)
        for dy in range(-remaining, remaining + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx <= 255 and 0 <= ny <= 255 and terrain.is_passable(nx, ny):
                return True
    return False


def select_new_combat_target(
    ctx: DecideCtx,
    threats: list[EnemyThreatDict],
) -> EnemyThreatDict | None:
    """Return the next viable new combat target.

    Picks the closest viable enemy that is not blocked or on kill
    cooldown and has reachable adjacent ground.

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
        and has_standoff_landing(threat["x"], threat["y"], ctx.terrain)
    ]
    if not viable:
        return None
    return viable[0]


def get_locked_target(
    ctx: DecideCtx,
    threats: list[EnemyThreatDict],
) -> EnemyThreatDict | None:
    """Return the lock target IFF the threat list still includes it.

    The threat list is the single source of truth for "this tank is
    viewport-confirmed right now". The pre-2026-06-21 implementation
    also fell back to the world-state tanks registry to synthesise
    a fake threat for tanks that had moved off-viewport, but the
    enemy-tracking probe proved that fallback was the second source
    of the "fires one shot then hops" loop: it kept the lock alive
    on tanks the JS client itself no longer listed in
    ``activeGame.P.j``, which sent the bot teleporting after
    phantoms. Now: no fallback. If the locked tank leaves the
    viewport, ``_decide_hunt_engage`` enters confirm_kill and the
    bot re-acquires from fresh intel.

    Args:
        ctx: Decision context.
        threats: Current threat list.

    Returns:
        The matching threat from ``threats``, or ``None``.
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
        ctx.timestamp_ms,
        ctx.ws,
    )


def _combat_landing_candidates(
    ctx: DecideCtx,
    target: EnemyThreatDict,
) -> list[tuple[int, int]]:
    """Return usable adjacent landing tiles ordered by distance to self."""
    return shared_combat_landing_candidates(
        ctx.filtered,
        ctx.self_state,
        target,
        ctx.terrain,
        ctx.timestamp_ms,
        ctx.ws,
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

    threats = analyze_threats(
        ctx.ws,
        ctx.filtered,
        ctx.self_state,
        ctx.timestamp_ms,
        human_min_rank=ctx.config["human_target_min_rank"],
        human_max_rank=ctx.config["human_target_max_rank"],
    )
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
            "find_target",
            AIStateDict(
                **{
                    **_set_combat_target(base_with_block, next_target),
                    "last_map_open_ms": ctx.timestamp_ms,
                }
            ),
            ctx.equip,
            reason_context={"target_name": next_target["name"]},
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


__all__ = [
    "block_combat_target_and_replan",
    "clear_combat_target",
    "combat_landing_tile",
    "get_locked_target",
    "has_clear_shot_line",
    "has_standoff_landing",
    "is_already_engaged",
    "select_new_combat_target",
]
