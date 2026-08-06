"""Teleport-based harvest steps of the COLLECT cascade.

The larder family (cascade step 5: known-stock hops before any
discovery) plus the marooned-ladder rungs (desperation hop and
walk-for-fuel last resort). In-viewport pickups live in
:mod:`collect_pickups`; lock continuation in :mod:`collect_locks`.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.collect_common import (
    COLLECT_SCORE,
    emit_hop_declined,
    is_container_blacklisted,
)
from tankpit_bot.bot.ai.context import DecideCtx, make_decision, set_resource_target
from tankpit_bot.bot.ai.equipment_search import find_all_tracked_equipment
from tankpit_bot.bot.ai.ferry_landing import find_ferry_boarding_tile
from tankpit_bot.bot.ai.intent import release_collect_plan
from tankpit_bot.bot.ai.larder import select_fuel_larder_hop
from tankpit_bot.bot.ai.mode_controller import hunt_entry_permitted, weapon_reserves_below_break
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.reachability import find_attainable_landing_tile
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.state.types import ContainerStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


def _equipment_hop_landing(
    ctx: DecideCtx,
    terrain: TerrainMapProtocol,
    container: ContainerStateDict,
) -> tuple[int, int] | None:
    """Resolve an equipment hop's landing: own ground, else a ferry.

    The ground landing must be ATTAINABLE, not merely legal: a known
    mine on the tile displaces the landing (session
    bot-20260805-173034: 1,068 hops re-aimed at one mined service
    tile, 534 displaced teleports, zero pickups). Mine-denied access
    is the clearance shot's job, which runs earlier in the cascade;
    this selector only aims where the tank will actually stand.

    Water-locked equipment gets the ferry boarding-tile fallback
    exactly as the fuel larder does ([[flag-triage-20260729]] F5).
    Radar-situation receipt (2026-07-30 session 7): ALL 8 tracked
    equipment containers were water drops with every neighbor
    impassable, so the radar-broke tank had no reachable restock at
    all until this fallback.

    Args:
        ctx: Decision context (ferry beliefs + clock).
        terrain: Composed decision terrain (caller-narrowed non-None).
        container: Tracked equipment container.

    Returns:
        Landing coordinates, or ``None`` when neither ground nor a
        fresh believed ferry serves the container.
    """
    landing = find_attainable_landing_tile(
        terrain, ctx.world["mines"], container["x"], container["y"]
    )
    if landing is not None:
        return landing
    return find_ferry_boarding_tile(
        ctx.world,
        terrain,
        container["x"],
        container["y"],
        ctx.timestamp_ms,
    )


def _equipment_hop_barred(ctx: DecideCtx, base_state: AIStateDict) -> bool:
    """Return True when no equipment hop may be scheduled this tick.

    Two bars: inventory already combat-ready (the hunt-entry bar for
    LOCKLESS collection), and the held-lock bar -- F21
    ([[flag-triage-20260729]], fixed 2026-07-31): during a HELD combat
    lock the entry bar must not schedule travel. With Yuppler locked
    and duals at 22/25 it pulled an 85-tile round trip to top three
    duals, gifting the human ~500 fuel and two free windows. Mid-fight
    equipment hops need a genuine weapon BREAK; near stock
    (in-viewport walk pickups, cascade step 3) is always taken.

    Args:
        ctx: Decision context.
        base_state: Base AI state carrying the combat lock.

    Returns:
        True when the hop step must decline.
    """
    if hunt_entry_permitted(ctx):
        return True
    if base_state["combat_target_id"] != -1 and not weapon_reserves_below_break(ctx):
        emit_hop_declined("equipment", held_lock=1)
        return True
    return False


def hop_toward_equipment(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Teleport toward tracked equipment the walk-pickup step cannot reach.

    ``find_nearest_equipment`` filters to walkable-from-here candidates
    inside the current viewport; step 3 of the cascade
    (``select_and_pickup_equipment``) handles those and runs FIRST.
    Every tracked container this step sees is therefore not
    walk-actionable this tick — equipment revealed in other viewports
    by prior radar or 0x5A patches, AND in-viewport equipment whose
    walk path is terrain-blocked (the 2026-07-30 flag-4 fix: the old
    external-only filter hid walk-blocked in-viewport containers from
    both steps).

    User contract (2026-07-06): while inventory is below combat-ready
    the bot must be actively topping up. This cascade step reaches
    into the whole-map equipment atlas and teleports to the nearest
    container whose landing tile is legal and whose teleport cost
    leaves ``engagement_reserve`` behind. The predicate matches the
    yield-to-hunt gate (Bug 0.4): if HUNT is permitted, no hop is
    needed.

    Stale-belief caveat: a container tracked from a scan minutes ago
    may have been picked up by another player and no wire signal
    confirms distant consumption. The pragmatic Phase 0 hop accepts
    that risk and pays the wasted-teleport cost if the container is
    gone. Wasted hops still refresh the viewport and reveal fresher
    intel on the way to combat-ready.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        Teleport decision to the cheapest tracked equipment container
        with a legal landing, or ``None`` when inventory is
        combat-ready, a combat lock is held with every reserve above
        its break threshold (F21: mid-fight top-ups only at a genuine
        weapon break), terrain is unknown, nothing is tracked, no
        candidate has a legal landing tile, or every affordable
        teleport would leave the engagement reserve.
    """
    if _equipment_hop_barred(ctx, base_state):
        return None
    terrain = ctx.terrain
    if terrain is None:
        return None
    candidates = find_all_tracked_equipment(ctx.world)
    if not candidates:
        emit_hop_declined("equipment", no_candidates=1)
        return None
    # No external-only filter (2026-07-30, flag-4 fix): this step runs
    # AFTER the walk-pickup step declined, so every tracked container
    # left — including IN-viewport ones whose walk path is blocked by
    # terrain — is teleport fair game. The pre-fix filter hid
    # in-viewport walk-blocked equipment from both steps: run
    # bot-20260730-000038 ticks 121-126 dot-hopped away from three
    # identified containers and paid a return trip for them later
    # ([[flag-triage-20260729]]).
    landing_reserve = ctx.config["engagement_fuel_budget"] + ctx.config["fuel_low_threshold"]
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    no_landing = 0
    reserve_blocked = 0
    own_ground = 0
    best_cost = -1
    best_landing_x = 0
    best_landing_y = 0
    best_container: ContainerStateDict | None = None
    for container in candidates:
        landing = _equipment_hop_landing(ctx, terrain, container)
        if landing is None:
            no_landing += 1
            continue
        landing_x, landing_y = landing
        if (landing_x, landing_y) == (sx, sy):
            # A teleport that does not move the tank is never travel
            # (s8-2, [[flag-triage-20260729]]: the escape landing
            # re-derived a hop TO THE TILE THE TANK STOOD ON and
            # burned a map-open tick). Ground the tank already owns
            # belongs to the pickup steps.
            own_ground += 1
            continue
        cost = teleport_cost(sx, sy, landing_x, landing_y)
        if ctx.fuel - cost < landing_reserve:
            reserve_blocked += 1
            continue
        if best_container is None or cost < best_cost:
            best_cost = cost
            best_landing_x = landing_x
            best_landing_y = landing_y
            best_container = container
    if best_container is None:
        emit_hop_declined(
            "equipment",
            candidates=len(candidates),
            no_landing=no_landing,
            reserve_blocked=reserve_blocked,
            own_ground=own_ground,
            fuel=ctx.fuel,
            landing_reserve=landing_reserve,
        )
        return None
    emit_ai(
        "equipment hop to (%d,%d) landing (%d,%d) cost=%d (dual=%d homing=%d radar=%d)",
        best_container["x"],
        best_container["y"],
        best_landing_x,
        best_landing_y,
        best_cost,
        ctx.inventory["dual_shots"]["count"],
        ctx.inventory["homing_shots"]["count"],
        ctx.inventory["extra_radars"]["count"],
    )
    emit_diagnostic(
        diagnostic_kind="hop_selected",
        hop_kind="equipment",
        target_x=best_container["x"],
        target_y=best_container["y"],
        landing_x=best_landing_x,
        landing_y=best_landing_y,
        cost=best_cost,
    )
    return make_decision(
        make_teleport_command(best_landing_x, best_landing_y),
        "COLLECT",
        COLLECT_SCORE,
        best_landing_x,
        best_landing_y,
        "equipment_hop",
        # Larder semantics (2026-07-27): the hop holds an equipment
        # lock so the landing tick dispatches the pickup directly --
        # the landing tile IS the container when passable and the
        # own-tile pickup is live-proven ([[equipment-system]]) --
        # and the landing radar is suppressed (harvest hops never
        # spend a scan, [[larder-plan]]).
        AIStateDict(
            **{
                **set_resource_target(
                    base_state,
                    "equipment",
                    best_container["x"],
                    best_container["y"],
                ),
                "suppress_landing_scan": True,
            }
        ),
        ctx.equip,
    )


def larder_harvest(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Cascade step 5: harvest known stock before any discovery.

    Equipment first (matching the walk-pickup order), then the scored
    fuel larder ([[larder-plan]]).

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        Larder hop decision, or ``None`` when the larder is empty or
        unprofitable and the tick belongs to discovery.
    """
    equipment_hop = hop_toward_equipment(ctx, base_state)
    if equipment_hop is not None:
        return equipment_hop
    return _hop_toward_fuel_larder(ctx, base_state)


def _hop_toward_fuel_larder(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Teleport to the best-scoring remembered fuel container.

    The larder plan's fuel harvest ([[larder-plan]], user rulings
    2026-07-27): before spending radar or blind dot hops, drain the
    stock the session already verified. Candidates are re-scored every
    tick by ``min(volume, deficit) / cost`` (no fixed errands); the
    landing tile is inside the server's fuel auto-pick reach so the
    stop is ~2 ticks with no pickup command. The hop holds a fuel
    lock on the target and suppresses the landing radar.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        Teleport decision to the winning container's landing tile, or
        ``None`` when the tank is at capacity, terrain is unknown, or
        every believed fuel container fails the physics gates (legal
        landing, reserve, net-positive gain).
    """
    if ctx.fuel >= fuel_capacity(ctx.self_state["rank"]):
        return None
    if ctx.terrain is None:
        return None
    selection = select_fuel_larder_hop(ctx, is_blacklisted=is_container_blacklisted)
    container = selection["container"]
    if container is None:
        if selection["candidates"] > 0:
            emit_hop_declined(
                "fuel_larder",
                candidates=selection["candidates"],
                too_close=selection["too_close"],
                no_landing=selection["no_landing"],
                reserve_blocked=selection["reserve_blocked"],
                unprofitable=selection["unprofitable"],
                dreg=selection["dreg"],
                ferry_served=selection["ferry_served"],
                fuel=ctx.fuel,
            )
        return None
    emit_ai(
        "fuel larder hop to (%d,%d) vol=%d landing (%d,%d) cost=%d (fuel=%d)",
        container["x"],
        container["y"],
        container["volume"],
        selection["landing_x"],
        selection["landing_y"],
        selection["cost"],
        ctx.fuel,
    )
    emit_diagnostic(
        diagnostic_kind="hop_selected",
        hop_kind="fuel_larder",
        target_x=container["x"],
        target_y=container["y"],
        landing_x=selection["landing_x"],
        landing_y=selection["landing_y"],
        cost=selection["cost"],
    )
    return make_decision(
        make_teleport_command(selection["landing_x"], selection["landing_y"]),
        "COLLECT",
        COLLECT_SCORE,
        selection["landing_x"],
        selection["landing_y"],
        "fuel_hop",
        AIStateDict(
            **{
                **set_resource_target(base_state, "fuel", container["x"], container["y"]),
                "suppress_landing_scan": True,
            }
        ),
        ctx.equip,
        reason_context={"volume": container["volume"]},
    )


def desperation_fuel_hop(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Hop to the cheapest believed fuel container when marooned.

    Reached only at critical fuel with every normal cascade step
    declined. The normal hop selectors serve DISCOVERY and refuse
    freshly-scanned ground and reserve-breaking costs; a marooned
    tank is not discovering, it is surviving. Run bot-20260728-093011
    sat at fuel 68 on a shore patch with 13 radar-verified fuel
    containers nearby -- 12 water-locked against walking, but the
    auto-pick law ([[fuel-system]]) credits a teleport landing ON or
    CARDINALLY ADJACENT to a fuel container, so a 2-tile hop to a
    shore landing tile refuels where no walk can. Gates here are
    physics only: an attainable landing (`find_attainable_landing_tile`,
    the shore-aware, displacement-proof helper) and cost within the
    remaining tank. The hop
    holds a fuel lock and suppresses the landing radar (larder
    semantics, [[larder-plan]]).

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        Teleport decision to the cheapest believed container's
        landing, or ``None`` when terrain is unknown or no believed
        container has an affordable legal landing.
    """
    terrain = ctx.terrain
    if terrain is None:
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    best_cost = 0
    best_landing_x = 0
    best_landing_y = 0
    best_container: ContainerStateDict | None = None
    for container in ctx.world["containers"].values():
        if not container["is_fuel"] or container["volume"] <= 0:
            continue
        if container["failed_pickups"] > 0:
            continue
        if is_container_blacklisted(container["x"], container["y"]):
            continue
        landing = find_attainable_landing_tile(
            terrain, ctx.world["mines"], container["x"], container["y"]
        )
        if landing is None:
            continue
        landing_x, landing_y = landing
        cost = teleport_cost(sx, sy, landing_x, landing_y)
        if cost <= 0 or cost > ctx.fuel:
            continue
        if best_container is None or cost < best_cost:
            best_cost = cost
            best_landing_x = landing_x
            best_landing_y = landing_y
            best_container = container
    if best_container is None:
        return None
    emit_ai(
        "marooned at fuel %d: desperation hop to fuel at (%d,%d) vol=%d landing (%d,%d) cost=%d",
        ctx.fuel,
        best_container["x"],
        best_container["y"],
        best_container["volume"],
        best_landing_x,
        best_landing_y,
        best_cost,
    )
    emit_diagnostic(
        diagnostic_kind="hop_selected",
        hop_kind="fuel_desperation",
        target_x=best_container["x"],
        target_y=best_container["y"],
        landing_x=best_landing_x,
        landing_y=best_landing_y,
        cost=best_cost,
    )
    return make_decision(
        make_teleport_command(best_landing_x, best_landing_y),
        "COLLECT",
        COLLECT_SCORE,
        best_landing_x,
        best_landing_y,
        "fuel_hop",
        AIStateDict(
            **{
                **set_resource_target(
                    base_state,
                    "fuel",
                    best_container["x"],
                    best_container["y"],
                ),
                "suppress_landing_scan": True,
            }
        ),
        ctx.equip,
        reason_context={"volume": best_container["volume"]},
    )


WALK_FOR_FUEL_MAX_TILES = 48
"""Farthest known fuel a marooned tank will walk toward (~96 s of
2 s-per-tile walking). Beyond this the out_of_fuel exit stands -- the
2026-07-25 exposure rule caps how long a broke tank crawls in the
open, even in the practice room where bots never initiate."""


def walk_for_fuel_last_resort(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Walk toward the nearest known fuel instead of exiting broke.

    The final rung before the ``out_of_fuel`` exit, reached only when
    every pickup, larder hop, forage step, and dot hop has declined at
    critical fuel. Walking is free at any fuel level (the density
    probe's marooned-recovery law, [[walk-mechanics]]), so a tank with
    known fuel in walking range is NOT actually stuck: runs
    bot-20260728-090813/-091209 exited at fuel 98/88 in a shore corner
    with the whole dot atlas 15+ unaffordable-teleport tiles away.
    Each tick walks one in-viewport leg toward the nearest candidate
    (map dot or believed container); arrival is handled by the normal
    cascade -- fresh ground re-enables forage, scans, and pickups.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        A one-leg walk decision, or ``None`` when no known fuel is
        inside the walk cap or no leg is walkable (the exit stands).
        The caller guarantees critical fuel -- the healthy-fuel tick
        resolved via the hunt handoff before this rung.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    candidates: list[tuple[int, int, int]] = []
    for dot_x, dot_y in ctx.map_fuel_dots:
        candidates.append((abs(dot_x - sx) + abs(dot_y - sy), dot_x, dot_y))
    for container in ctx.world["containers"].values():
        if not container["is_fuel"] or container["volume"] <= 0:
            continue
        if container["failed_pickups"] > 0:
            continue
        if is_container_blacklisted(container["x"], container["y"]):
            continue
        candidates.append(
            (abs(container["x"] - sx) + abs(container["y"] - sy), container["x"], container["y"])
        )
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    # Nearest-first over EVERY candidate inside the cap: in a shore
    # corner the closest entries are water-locked containers whose leg
    # resolves to a teleport fallback, not a walk (run
    # bot-20260728-092357 gave up after trying only the nearest and
    # exited with dots in walking range further down the list).
    for _, target_x, target_y in sorted(
        c for c in candidates if 0 < c[0] <= WALK_FOR_FUEL_MAX_TILES
    ):
        terrain = ctx.terrain
        if terrain is not None and not terrain.is_passable(target_x, target_y):
            continue
        leg_x = min(max(target_x, left), right)
        leg_y = min(max(target_y, top), bottom)
        if (leg_x, leg_y) == (sx, sy):
            continue
        command = walk_or_teleport(ctx, leg_x, leg_y, pickup_kind=None)
        if command is None or command["cmd_type"] != "move":
            continue
        emit_ai(
            "marooned at fuel %d: walking leg (%d,%d) toward known fuel at (%d,%d)",
            ctx.fuel,
            leg_x,
            leg_y,
            target_x,
            target_y,
        )
        emit_diagnostic(
            diagnostic_kind="walk_for_fuel",
            target_x=target_x,
            target_y=target_y,
            leg_x=leg_x,
            leg_y=leg_y,
            fuel=ctx.fuel,
        )
        return make_decision(
            command,
            "COLLECT",
            COLLECT_SCORE,
            leg_x,
            leg_y,
            "walk_for_fuel",
            release_collect_plan(base_state, reason="walk_for_fuel_override"),
            ctx.equip,
        )
    return None


__all__ = [
    "WALK_FOR_FUEL_MAX_TILES",
    "desperation_fuel_hop",
    "hop_toward_equipment",
    "larder_harvest",
    "walk_for_fuel_last_resort",
]
