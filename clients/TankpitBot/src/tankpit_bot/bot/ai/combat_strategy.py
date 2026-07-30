"""Combat route primitives for the durable HUNT owner.

This module owns typed helper functions for target acquisition, teleport
landing, shoot/miss cycles, and blocked-target replanning. Top-level owner
selection now lives in ``ai_strategy`` and ``hunt_mode``.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.combat_landing import (
    choose_combat_landing_tile,
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
    BotCommand,
    make_map_open_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.sniffer.world_state import is_move_target_failed
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


def is_already_engaged(ctx: DecideCtx) -> bool:
    """Return True when the bot has already dispatched a shot at the locked target.

    The discriminator is ``last_shot_target_id`` -- set only when
    :func:`_combat_shoot` actually dispatches a ``shoot`` command for
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


def has_passable_adjacent(
    x: int,
    y: int,
    terrain: TerrainMapProtocol | None,
) -> bool:
    """Return True when at least one cardinal neighbor is passable ground."""
    if terrain is None:
        return True
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
        and has_passable_adjacent(threat["x"], threat["y"], ctx.terrain)
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


# =============================================================================
# Internal helpers
# =============================================================================


def _has_damaging_weapon_available(ctx: DecideCtx) -> bool:
    """Return True when at least one damaging weapon slot can fire this tick.

    A "damaging" slot is a dual or homing that is both enabled and
    stocked. Radars, missiles, and shields don't damage tanks; single
    (weapon=0) is what the server picks as the fallback when neither
    dual nor homing is available. The predicate distinguishes the two
    causes of a ``weapon=0`` miss:

    * ``afterimage_confirmed`` -- dual OR homing was available and the
      server still picked single, so the target is not at the aim tile.
    * ``ammo_exhaustion_miss`` -- neither dual nor homing was
      available, so the server routed to single by default. The miss
      says nothing about the target's presence.

    Used by the stationary-miss classifier in :func:`_combat_shoot`
    (Bug 0.6): only the ``afterimage_confirmed`` case warrants
    blacklisting the target.

    Args:
        ctx: Decision context.

    Returns:
        True when ``dual_shots`` or ``homing_shots`` has both
        ``enabled=True`` and ``count > 0``.
    """
    duals = ctx.inventory["dual_shots"]
    homings = ctx.inventory["homing_shots"]
    return (duals["enabled"] and duals["count"] > 0) or (
        homings["enabled"] and homings["count"] > 0
    )


def refuel_for_hunt(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Delegate the tick to the fuel planner when hunting is fuel-starved.

    Threats sort nearest-first and teleport cost is monotone in
    distance, so an unaffordable nearest target means EVERY target is
    unaffordable. Blocking and replanning instead cascaded through the
    whole roster and ended in a map-reopen spin: run 20260611-025636
    spawned at fuel 620 -- above the fuel-low entry rule (500) but
    below every engagement's cost-plus-reserve -- and spent its entire
    240s on 115 map reopens without a single shot. Collecting fuel is
    the only decision that changes the blocked condition.

    Refuel-then-RESUME (user ruling 2026-07-27, closing the last
    voluntary live-target drop): the lock is KEPT through the fuel
    detour, so the 2026-07-25 resume machinery returns to this exact
    target once the tank can fund the trip -- run 183703's red-1 was
    deferred at fuel 239 under the old clear-and-reacquire and never
    hunted again (the guest won the fresh distance race). The old
    anti-spin property survives because each deferred tick delegates
    to a real collect decision (fuel strictly grows), not a map
    reopen.

    When the collect cascade itself declines (fuel healthy but nothing
    collectible in reach), the fuel situation is not going to improve
    this tick, so the unaffordable target is blocked and replanned
    instead of exiting the session -- the terminator that also bounds
    the pathological corner where cost + reserve exceeds the tank's
    own fuel capacity.

    Args:
        ctx: Decision context.
        target: The unaffordable combat target being deferred.

    Returns:
        Fuel recovery decision with combat target cleared, or a
        blocked-target replanning decision when collection declines.
    """
    # Lazy import: collect_mode imports clear_combat_target from
    # this module at import time.
    from tankpit_bot.bot.ai.collect_mode import decide_collect_mode

    locked_ctx = DecideCtx(
        ctx.world,
        ctx.self_state,
        _set_combat_target(ctx.base, target),
        ctx.inventory,
        ctx.timestamp_ms,
        ctx.terrain,
        ctx.combat_feedback,
        ctx.map_fuel_dots,
    )
    decision = decide_collect_mode(locked_ctx)
    if decision is None:
        emit_ai(
            "refuel-for-hunt found nothing collectible at fuel %d, blocking %s",
            ctx.fuel,
            target["name"],
        )
        return block_combat_target_and_replan(ctx, target)
    return decision


def _combat_open_map(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase 0: Open map to get fresh enemy positions."""
    emit_ai("open map to find %s", target["name"])
    return make_decision(
        make_map_open_command(),
        "HUNT",
        800,
        0,
        0,
        "find_target",
        AIStateDict(
            **{
                **_set_combat_target(ctx.base, target),
                "last_map_open_ms": ctx.timestamp_ms,
            }
        ),
        ctx.equip,
        reason_context={"target_name": target["name"]},
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


def _combat_teleport(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase 1: Teleport to enemy."""
    landing_x, landing_y = combat_landing_tile(ctx, target)
    if is_move_target_failed(landing_x, landing_y, ctx.timestamp_ms):
        emit_ai(
            "combat landing (%d,%d) for %s already failed, blocking target",
            landing_x,
            landing_y,
            target["name"],
        )
        return block_combat_target_and_replan(ctx, target)
    # Engaging must leave fuel to FIGHT, not merely to exist: the
    # reserve is fuel_low_threshold (the line where COLLECT outranks
    # HUNT) plus the full engagement_fuel_budget -- the same
    # end-to-end funding the acquisition gate demands, so a chase
    # re-teleport can never slip through cheaper than the fight it
    # funds. Two priced incidents behind the sum: run 20260611-004505
    # (a teleport gated only on hunt_min_fuel landed at 224, the fuel
    # mode hijacked the next tick, and the ~190 spent reaching
    # purple-8 bought a forbidden fight) and run 20260729-105325 (the
    # fuel_low_threshold-only reserve passed a 158-cost chase at 372
    # by 14 fuel, landed at 214, LOW_FUEL hijacked one shot later,
    # session min 140 -- user ruling 2026-07-29: "we cant kill anyone
    # if we die... we should fuel before chasing"). Refuel-for-hunt
    # keeps the lock and prefers in-viewport walk pickups, then the
    # larder, so the detour is usually one container away.
    engagement_reserve = ctx.config["fuel_low_threshold"] + ctx.config["engagement_fuel_budget"]
    if not can_afford_teleport(
        ctx,
        landing_x,
        landing_y,
        reserve_fuel=engagement_reserve,
    ):
        emit_ai(
            "cannot afford combat teleport for %s to (%d,%d) (fuel=%d cost=%d reserve=%d)"
            " - refueling before hunt",
            target["name"],
            landing_x,
            landing_y,
            ctx.fuel,
            teleport_fuel_cost_to(ctx, landing_x, landing_y),
            engagement_reserve,
        )
        return refuel_for_hunt(ctx, target)
    emit_ai(
        "teleport near %s to (%d,%d) (target at %d,%d)",
        target["name"],
        landing_x,
        landing_y,
        target["x"],
        target["y"],
    )
    return make_decision(
        make_teleport_command(landing_x, landing_y),
        "HUNT",
        800,
        landing_x,
        landing_y,
        "teleport_target",
        _set_combat_target(ctx.base, target),
        ctx.equip,
        reason_context={"target_name": target["name"]},
    )


def teleport_to_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Teleport toward the given combat target when legal.

    Args:
        ctx: Decision context.
        target: Combat target to close on.

    Returns:
        Teleport decision, or a blocked-target replanning decision when the
        landing tile is unusable or the teleport is unaffordable.
    """
    return _combat_teleport(ctx, target)


def _combat_close(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Phase closing: shoot when in range or already engaged; teleport on first close.

    User-contract gameplay loop (2026-06-26): open map, teleport
    cardinally adjacent to the target, dual until the target teleports
    away, then stay in place and fire homing until the target is
    deactivated. Enemies don't move *within* the viewport -- when they
    leave cardinal adjacency, they teleported -- so chasing them with
    another teleport burns fuel without changing the firing geometry.

    Three branches:

    1. **Cardinally adjacent** -- shoot (server picks ``dual`` for the
       guaranteed point-blank hit).
    2. **Already engaged** (:func:`is_already_engaged`) -- shoot from
       the current tile. The server picks ``homing`` when the target
       is not adjacent and homing tracks, so a stay-put fire continues
       to land hits without spending fuel on another teleport.
    3. **Fresh acquire, not adjacent** -- teleport directly to the
       target. This is the one-time close the engagement contract
       allows; subsequent ticks fall through branch (2).
    """
    if has_cardinal_combat_shot(ctx.self_state, target):
        return _combat_shoot(ctx, target)
    dist = abs(ctx.self_state["x"] - target["x"]) + abs(ctx.self_state["y"] - target["y"])
    if is_already_engaged(ctx):
        emit_ai(
            "engaged %s moved off cardinal from (%d,%d) target=(%d,%d) dist=%d; staying put",
            target["name"],
            ctx.self_state["x"],
            ctx.self_state["y"],
            target["x"],
            target["y"],
            dist,
        )
        return _combat_shoot(ctx, target)
    emit_ai(
        "fresh acquire of %s from (%d,%d) target=(%d,%d) dist=%d; teleporting to close",
        target["name"],
        ctx.self_state["x"],
        ctx.self_state["y"],
        target["x"],
        target["y"],
        dist,
    )
    return _combat_teleport(ctx, target)


def close_target(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
    """Close distance on the given combat target.

    Args:
        ctx: Decision context.
        target: Combat target to approach.

    Returns:
        Close-range combat decision: a shot when already in cardinal position,
        a teleport when affordable, or a blocked-target replanning decision.
    """
    return _combat_close(ctx, target)


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
        if container is not None:
            x, y = container["x"], container["y"]
            if want_fuel:
                return make_pickup_fuel_command(x, y)
            return make_pickup_equipment_command(x, y)
    return None


def _combat_shoot(ctx: DecideCtx, target: EnemyThreatDict) -> TickDecisionDict:
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


# The server's effective shot range, measured across 350 shots on
# 2026-06-11: Manhattan distance 1 hit 255/255, distance 2 hit 1/1,
# distance 4+ hit ~0% (45 misses at 4, 35 at 12; the two distance-15
# "hits" were homing shots, which track). Shots in range never miss --
# and the range is adjacency, not the awareness-range combat_range.
SHOT_RANGE_TILES = 2


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


__all__ = [
    "SHOT_RANGE_TILES",
    "block_combat_target_and_replan",
    "clear_combat_target",
    "close_target",
    "combat_landing_tile",
    "engage_target",
    "get_locked_target",
    "has_cardinal_combat_shot",
    "has_combat_shot",
    "has_passable_adjacent",
    "is_already_engaged",
    "open_map_for_target",
    "select_new_combat_target",
    "teleport_to_target",
]
