"""Teleport-based harvest steps of the COLLECT cascade.

The larder family (cascade step 5: known-stock hops before any
discovery) plus the marooned desperation hop. The marooned
walk-and-pan last resort lives in :mod:`maroon_walk`; in-viewport
pickups in :mod:`collect_pickups`; lock continuation in
:mod:`collect_locks`.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.collect_common import (
    COLLECT_SCORE,
    emit_hop_declined,
    split_fresh_hop_sightings,
)
from tankpit_bot.bot.ai.context import DecideCtx, make_decision
from tankpit_bot.bot.ai.equipment_search import find_all_tracked_equipment
from tankpit_bot.bot.ai.ferry import SurfaceRouteTerrain, is_riding_ferry
from tankpit_bot.bot.ai.ferry_landing import find_ferry_boarding_tile
from tankpit_bot.bot.ai.intent import set_resource_target
from tankpit_bot.bot.ai.larder import WALK_DOMINANT_RANGE, select_fuel_larder_hop
from tankpit_bot.bot.ai.mode_gates import (
    hunt_entry_permitted,
    weapon_reserves_below_break,
)
from tankpit_bot.bot.ai.reachability import (
    find_attainable_landing_tile,
    is_collection_reachable_in_viewport,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_teleport_command
from tankpit_bot.inventory import inventory_counts
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.physics.supervisor import equipment_pickup_refusal
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
)
from tankpit_bot.state.types import ContainerStateDict


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
    landing = find_attainable_landing_tile(terrain, container["x"], container["y"])
    if landing is not None:
        return landing
    boarding = find_ferry_boarding_tile(
        ctx.world,
        terrain,
        container["x"],
        container["y"],
    )
    if boarding is not None:
        sx, sy = ctx.self_state["x"], ctx.self_state["y"]
        if max(abs(sx - boarding[0]), abs(sy - boarding[1])) <= 1:
            # Standing at the boarding tile and still deriving the
            # boarding hop IS the ride-failed receipt (the islet
            # loop, 2026-08-26) — the candidate is dead until the
            # tank moves away. Mirrors the fuel larder's ride_dead
            # gate.
            return None
    return boarding


def _equipment_hop_barred(ctx: DecideCtx, base_state: AIStateDict) -> bool:
    """Return True when no equipment hop may be scheduled this tick.

    Three bars: every slot at rank cap (the physics bar -- the shared
    ``equipment_pickup_refusal`` law; added 2026-08-20 after the
    gatherer livelock), inventory already combat-ready (the hunt-entry
    bar for LOCKLESS collection), and the held-lock bar -- F21
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
    refusal = equipment_pickup_refusal(inventory_counts(ctx.inventory), ctx.self_state["rank"])
    if refusal is not None:
        # The physics bar, in addition to the doctrinal one below: a
        # tank with every slot at rank cap gains nothing from any
        # equipment container, so a hop can only burn fuel on a
        # guaranteed code-7 refusal. For a FIGHTER the hunt-entry bar
        # below always fires first (full stock permits hunting), which
        # is why this bar was invisibly absent until the gatherer role
        # made ``hunt_entry_permitted`` unconditionally False and a
        # full recruit hopped at equipment it could not hold
        # (bot-20260820-005115, the full-inventory livelock's entry
        # teleport).
        emit_hop_declined("equipment", at_capacity=1)
        return True
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

    Stale-belief law (2026-09-01, superseding the Phase 0
    accept-the-risk stance): no wire signal confirms distant
    consumption, and in a co-farmed room the phantom rate proved the
    old caveat wrong — run bot-20260901-033100 paid eleven code-4
    empties in 2.5 minutes hopping to aged sightings while its
    zero-radar sweep starved. Sightings older than the hop pricing
    horizon (:data:`~tankpit_bot.bot.ai.collect_common.HOP_SIGHTING_MAX_AGE_MS`)
    are skipped by this lane; the beliefs survive for walk service
    and in-window pickups, and the declined tick falls through to
    sweep/frontier discovery.

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
    landing_reserve = ctx.engagement_budget + ctx.fuel_low_floor
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    no_landing = 0
    reserve_blocked = 0
    own_ground = 0
    # Hop pricing horizon (run bot-20260901-033100: eleven code-4
    # empties in 2.5 minutes against aged own and fleet-merged
    # sightings). The beliefs themselves survive — walk service and
    # in-window pickups still use them.
    fresh_candidates, stale = split_fresh_hop_sightings(candidates, ctx.timestamp_ms)
    best_cost = -1
    best_landing_x = 0
    best_landing_y = 0
    best_container: ContainerStateDict | None = None
    for container in fresh_candidates:
        landing = _equipment_hop_landing(ctx, terrain, container)
        if landing is None:
            no_landing += 1
            continue
        landing_x, landing_y = landing
        if abs(landing_x - sx) + abs(landing_y - sy) <= WALK_DOMINANT_RANGE and (
            is_collection_reachable_in_viewport(
                ctx.world,
                SurfaceRouteTerrain(terrain, water=is_riding_ferry(ctx.world)),
                sx,
                sy,
                container["x"],
                container["y"],
            )
        ):
            # A teleport inside the walk-dominant range is never
            # travel. Distance 0 was s8-2 ([[flag-triage-20260729]]:
            # the escape landing re-derived a hop TO THE TILE THE
            # TANK STOOD ON and burned a map-open tick); HUD flag 1
            # (2026-08-13 20:47) extended it -- under fire the hop
            # lane paid a map open plus a cost-6 teleport for a
            # container ONE TILE away that the ordinary pickup served
            # four seconds later. Ground within walking reach belongs
            # to the pickup steps and the clearance shot, exactly as
            # the larder's walk-territory rule already says for fuel
            # -- but ONLY when the walk lane can actually take it
            # (the same reachability predicate the pickup dispatch
            # uses). Deferring by distance alone re-opened the
            # 2026-07-30 flag-4 gap one gate lower and left rock-
            # pocketed near stock with nobody serving it
            # ([[flag-triage-20260902]]).
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
            stale=stale,
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
    selection = select_fuel_larder_hop(ctx)
    container = selection["container"]
    if container is None:
        if selection["candidates"] > 0:
            emit_hop_declined(
                "fuel_larder",
                candidates=selection["candidates"],
                too_close=selection["too_close"],
                stale=selection["stale"],
                no_landing=selection["no_landing"],
                reserve_blocked=selection["reserve_blocked"],
                unprofitable=selection["unprofitable"],
                dreg=selection["dreg"],
                ferry_served=selection["ferry_served"],
                ride_dead=selection["ride_dead"],
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
        landing = find_attainable_landing_tile(terrain, container["x"], container["y"])
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


__all__ = [
    "desperation_fuel_hop",
    "hop_toward_equipment",
    "larder_harvest",
]
