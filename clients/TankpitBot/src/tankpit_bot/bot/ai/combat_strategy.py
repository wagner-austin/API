"""Combat fire decision: aim, pickup-of-opportunity, and engage.

The chokepoint every shoot path reaches. Reads
:mod:`tankpit_bot.bot.ai.combat_prep` for the refuel/refresh stages it
falls back to, and :mod:`tankpit_bot.bot.ai.combat_target` for the
lock. The approach stages that call INTO this module live in
:mod:`tankpit_bot.bot.ai.combat_close`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_landing import (
    SHOT_RANGE_TILES,
)
from tankpit_bot.bot.ai.combat_prep import (
    _has_damaging_weapon_available,
    open_map_for_target,
    refuel_for_hunt,
)
from tankpit_bot.bot.ai.combat_target import (
    _set_combat_target,
    block_combat_target_and_replan,
)
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    make_decision,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_shoot_command,
)
from tankpit_bot.inventory import inventory_counts
from tankpit_bot.physics.supervisor import equipment_pickup_refusal, fuel_pickup_refusal
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
)
from tankpit_bot.state.types import SelfStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


def _clamp_aim_into_viewport(ctx: DecideCtx, aim_x: int, aim_y: int) -> tuple[int, int]:
    """Clamp a shoot aim onto the visible viewport.

    The server rejects any ``shoot`` whose aim tile is outside the
    16x16 visible viewport with 0x52 code 0 ("You can't do this") —
    live run 2026-07-03 20:34 drew five such rejections aiming at a
    pursuit target 5 rows below the viewport. The aim is only a hint:
    the wire-proven snipe pattern fires at an in-viewport ground tile
    with the target's ``tank_id`` and the server picks ``homing``,
    whose seeker tracks the real target (same run: ``weapon=3`` hit
    from an aim at the target's vacated tile). Off-viewport registry
    coordinates leak into pursuit aims because 0x3D MovementResponse
    broadcasts every map tank's position ~every 2 s — clamping the
    dispatched aim keeps every shot legal without touching the
    registry truth.

    The clamp only applies when the recorded viewport contains the
    bot's own tank — the tank is always inside its actual viewport, so
    a record that excludes it is stale or not yet established (the
    origin arrives with the landing 0x5A) and clamping against it
    would aim at garbage.

    Args:
        ctx: Decision context.
        aim_x: Desired aim X (may be outside the viewport).
        aim_y: Desired aim Y (may be outside the viewport).

    Returns:
        The aim clamped into the visible viewport bounds, or unchanged
        when the viewport record does not contain the bot.
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    self_x, self_y = ctx.self_state["x"], ctx.self_state["y"]
    if not (left <= self_x <= right and top <= self_y <= bottom):
        return (aim_x, aim_y)
    return (max(left, min(right, aim_x)), max(top, min(bottom, aim_y)))


def has_combat_shot(ctx: DecideCtx, target: EnemyThreatDict) -> bool:
    """Return True when the target is within the server's shot range.

    Args:
        ctx: Decision context (unused fields reserved; kept so every
            engagement predicate shares one signature).
        target: Enemy threat with its current-tick distance.

    Returns:
        True if the target's Manhattan distance is within
        ``SHOT_RANGE_TILES``.
    """
    del ctx
    return target["distance"] <= SHOT_RANGE_TILES


def has_cardinal_combat_shot(
    self_state: SelfStateDict,
    target: EnemyThreatDict,
) -> bool:
    """Return True when self is cardinally adjacent to the target.

    Cardinal adjacency (Manhattan distance exactly 1) is the geometry
    required for a guaranteed hit at point-blank range.

    Args:
        self_state: Player's own state.
        target: Enemy threat.

    Returns:
        True if Manhattan distance is exactly 1.
    """
    return abs(self_state["x"] - target["x"]) + abs(self_state["y"] - target["y"]) == 1


def _find_combat_pickup(ctx: DecideCtx) -> BotCommand | None:
    """Find an adjacent container to grab between shots.

    Checks for both fuel and equipment within one tile. Fuel is preferred
    when below the low threshold; equipment otherwise.

    Args:
        ctx: Decision context with world state and self position.

    Returns:
        A pickup command for the adjacent container, or None.
    """
    from tankpit_bot.bot.ai.equipment_search import find_adjacent_container

    self_state = ctx.world["self_state"]
    if self_state is None:
        return None

    fuel_low = self_state["fuel"] < ctx.config["fuel_low_threshold"]
    for want_fuel in [True, False] if fuel_low else [False, True]:
        container = find_adjacent_container(
            ctx.world,
            self_state,
            ctx.terrain,
            want_fuel=want_fuel,
        )
        if container is None:
            continue
        # The shared 0x52 refusal laws (physics/supervisor.py): a sip
        # at rank fuel capacity or a grab with every slot at the rank
        # cap transfers nothing — skip to the other kind instead of
        # burning the dispatch (48 refused sips in the 20-kill soak
        # bot-20260802-205105, every one 2s after a kill at 1100/1100).
        if want_fuel:
            refusal = fuel_pickup_refusal(
                self_state["fuel"], self_state["rank"], container["volume"]
            )
        else:
            refusal = equipment_pickup_refusal(inventory_counts(ctx.inventory), self_state["rank"])
        if refusal is not None:
            continue
        x, y = container["x"], container["y"]
        if want_fuel:
            return make_pickup_fuel_command(x, y)
        return make_pickup_equipment_command(x, y)
    return None


def engage_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase engaging: shoot; a miss on a STATIONARY target blocks it.

    This is the single chokepoint every shoot path reaches (direct
    engage, close-in shot, refresh-then-engage, locked-target pursuit).

    Wire-silence is **not** a stop signal. A locked target that
    teleports off the bot's viewport stops broadcasting wire updates
    -- the server only emits wire events for tanks the local viewport
    can see -- which is the expected case the pursuit cascade exists
    to handle. The lock holds until an authoritative deactivation
    signal arrives (``liveness`` flips to ``deactivated`` or the
    tank lands in ``killed_tank_ids``); pursuit fires homing toward
    the last wire position until then, and the server picks ``homing``
    when the target is mid-move or out of point-blank range because
    homing tracks.

    Live adjacent targets hit 255/255 (2026-06-11 data), so a shot
    that comes back with no damage against a target that has not moved
    proves the target is not killable right now -- a corpse from an
    unwitnessed kill or a shielded tank. Reopening the map (the old
    miss response) changes nothing about a visible target: run
    20260611-103244 shot the same frozen tile 12 times in a row, each
    miss buying a 2s map reopen. Blocking uses the same cooldown as
    kills, so a shielded tank gets retried after its shields are
    plausibly down. A miss against a target that MOVED since the shot
    is the one ambiguous case -- a live enemy may simply have stepped
    off the tile as the shot resolved -- so a mover is re-aimed at its
    fresh registry position instead of being abandoned.


    Args:
        ctx: Decision context.
        target: Combat target to shoot at.

    Returns:
        Combat engage decision, including miss-driven refresh behavior.
    """
    if ctx.combat_feedback == "rejected":
        # The server refused the previous dispatch outright (0x52
        # code 0/3/8) -- no ShootEvent, no ammo delta. With the aim
        # clamp below every dispatch is viewport-legal, so a residual
        # rejection means the server refuses this engagement geometry
        # for a reason the bot cannot see; repeating the identical
        # shot cannot change the answer (live run 2026-07-03 20:34:
        # five identical redispatches, 4 s of dead wait each).
        emit_ai(
            "server rejected shot at %s (%d,%d) - blocking target",
            target["name"],
            target["x"],
            target["y"],
        )
        return block_combat_target_and_replan(ctx, target)

    if ctx.combat_feedback == "miss":
        last_shot_at = (ctx.ai_state["combat_target_x"], ctx.ai_state["combat_target_y"])
        target_stationary = (target["x"], target["y"]) == last_shot_at
        dist = abs(ctx.self_state["x"] - target["x"]) + abs(ctx.self_state["y"] - target["y"])
        emit_diagnostic(
            diagnostic_kind="combat_miss",
            target_name=target["name"],
            target_id=target["tank_id"],
            shot_x=last_shot_at[0],
            shot_y=last_shot_at[1],
            current_x=target["x"],
            current_y=target["y"],
            self_x=ctx.self_state["x"],
            self_y=ctx.self_state["y"],
            dist=dist,
            target_moved=not target_stationary,
        )
        if target_stationary:
            # A consumption-miss (weapon=0: the server spent nothing and
            # resolved the shot against empty ground) at a registry
            # position that has not moved usually means the target is
            # NOT there -- a frozen registry entry after the target
            # left, or a corpse from an unwitnessed kill. Repeating
            # the shot cannot change the answer (live run 2026-07-02
            # 01:23: 25+ weapon=0 shots at orange-1's stale tile in a
            # 2s loop). Contract 3.3: miss on a stationary target
            # blocks it.
            #
            # Bug 0.6 (2026-07-06 22:39/22:40): the same weapon=0 +
            # stationary miss ALSO fires when the bot has no damaging
            # weapon left (duals + homings both exhausted or disabled),
            # because the server routes to single as the only remaining
            # option. That is an ``ammo_exhaustion_miss``, not an
            # ``afterimage_confirmed`` miss -- the target is live, the
            # bot just cannot damage it right now. Blacklisting a live
            # target in that case is wrong; the right response is to
            # disengage and refill.
            if _has_damaging_weapon_available(ctx):
                # Departed, not "unkillable": corpses and shields
                # return positive tile echoes but a consumption-miss
                # means the reroute window closed on an escaped
                # target ([[shoot-event-format]]#reroute-ttl-ms; run
                # 194658 confirmed the wall live: hits to +12.0 s,
                # miss at +14.0 s). User ruling 2026-07-26: live
                # targets are NEVER dropped -- hold the lock and open
                # the map; the resume machinery in HUNT/ACQUIRE
                # (_locked_target_pursuit) chases the refreshed
                # position, and it alone releases when the target is
                # genuinely dead or gone from the registry (the
                # original orange-2 case, user ruling 2026-07-20).
                emit_diagnostic(
                    diagnostic_kind="target_chase",
                    target_id=target["tank_id"],
                    target_name=target["name"],
                    last_x=target["x"],
                    last_y=target["y"],
                )
                emit_ai(
                    "miss on stationary %s at (%d,%d) - reroute window closed; "
                    "holding lock and chasing via map",
                    target["name"],
                    target["x"],
                    target["y"],
                )
                return open_map_for_target(ctx, target)
            emit_ai(
                "miss on stationary %s at (%d,%d) but no damaging weapon available - "
                "disengaging to refill (ammo exhaustion, not afterimage)",
                target["name"],
                target["x"],
                target["y"],
            )
            return refuel_for_hunt(ctx, target)
        emit_ai(
            "miss on %s at (%d,%d) dist=%d moved=True - re-aiming",
            target["name"],
            target["x"],
            target["y"],
            dist,
        )

    aim_x, aim_y = _clamp_aim_into_viewport(ctx, target["x"], target["y"])
    if (aim_x, aim_y) != (target["x"], target["y"]):
        emit_ai(
            "pursuit aim (%d,%d) is outside the viewport - clamped to (%d,%d); "
            "server homing tracks %s from the legal tile",
            target["x"],
            target["y"],
            aim_x,
            aim_y,
            target["name"],
        )
    emit_ai("shoot %s at (%d,%d)", target["name"], aim_x, aim_y)
    engaging_state = _set_combat_target(ctx.base, target)
    secondary = _find_combat_pickup(ctx)
    if secondary is not None:
        emit_ai("mid-combat pickup %s", secondary["cmd_type"])
    return make_decision(
        make_shoot_command(aim_x, aim_y, target["tank_id"]),
        "HUNT",
        800,
        aim_x,
        aim_y,
        "shoot_target",
        AIStateDict(
            **{
                **engaging_state,
                "last_shoot_ms": ctx.timestamp_ms,
                "last_shot_target_id": target["tank_id"],
                "last_shot_target_name": target["name"],
            }
        ),
        ctx.equip,
        reason_context={"target_name": target["name"]},
        secondary_command=secondary,
    )


__all__ = [
    "engage_target",
    "has_cardinal_combat_shot",
    "has_combat_shot",
]
