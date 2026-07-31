"""Durable COLLECT owner: unified fuel-and-equipment recovery cascade.

The user's gameplay loop is one mode: drain the equipment in the
current viewport, drain the fuel (when below cap), radar/walk to expand
coverage, and teleport to a fresh viewport when nothing actionable
remains here.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_break import INCOMING_RATE_WINDOW_MS
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    locked_resource_target,
    make_decision,
    set_resource_target,
)
from tankpit_bot.bot.ai.equipment import (
    is_fuel_lock_release_warranted,
    is_lock_release_warranted,
)
from tankpit_bot.bot.ai.equipment_search import (
    describe_container_search,
    find_all_tracked_equipment,
    find_best_fuel,
    find_equipment_candidates,
    find_nearest_equipment,
    find_teleport_landing_tile,
)
from tankpit_bot.bot.ai.forage import plan_forage_search
from tankpit_bot.bot.ai.intent import (
    current_collect_plan,
    plan_completes_here,
    release_collect_plan,
)
from tankpit_bot.bot.ai.larder import select_fuel_larder_hop
from tankpit_bot.bot.ai.mine_clearance import find_mine_clearance_shot
from tankpit_bot.bot.ai.mode_controller import hunt_entry_permitted
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.resource_search import (
    make_resource_search_hop,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.inventory import inventory_all_full
from tankpit_bot.physics.capacity import fuel_capacity, inventory_capacity
from tankpit_bot.physics.costs import teleport_cost
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.sniffer.world_state import (
    get_incoming_damage_window,
    is_move_target_failed,
    recent_movement_rejections,
)
from tankpit_bot.state.types import ContainerStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

_COLLECT_SCORE = 925

_blacklisted_container_keys: set[str] = set()


def _blacklist_container(x: int, y: int) -> None:
    key = f"{x},{y}"
    if key not in _blacklisted_container_keys:
        emit_diagnostic(diagnostic_kind="container_blacklisted", x=x, y=y)
    _blacklisted_container_keys.add(key)


def is_container_blacklisted(x: int, y: int) -> bool:
    """Check if a container is permanently blacklisted.

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.

    Returns:
        True if the container has been blacklisted this session.
    """
    return f"{x},{y}" in _blacklisted_container_keys


def reset_container_blacklist() -> None:
    """Clear the container blacklist (called on death/respawn)."""
    _blacklisted_container_keys.clear()


def select_equipment_target(
    ctx: DecideCtx,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the nearest executable walk-reachable equipment target.

    Targets with a live teleport-approach mark are skipped: a marked
    container already ate a teleport without becoming collectable.

    Args:
        ctx: Decision context.

    Returns:
        ``(container, command)`` for the nearest executable equipment target, or
        ``None`` when no visible equipment target can currently be executed.
    """
    candidates = [
        container
        for container in find_equipment_candidates(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
        )
        if not is_container_blacklisted(container["x"], container["y"])
    ]
    if not candidates:
        return None

    container = candidates[0]
    command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="equipment")
    if command is None:
        return None
    return (container, command)


def select_fuel_target(
    ctx: DecideCtx,
) -> tuple[ContainerStateDict, BotCommand] | None:
    """Return the best executable walk-reachable fuel target.

    Args:
        ctx: Decision context.

    Returns:
        ``(container, command)`` for the best executable fuel target, or
        ``None`` when no visible fuel target can currently be executed.
    """
    container = find_best_fuel(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        minimum_volume=1,
    )
    if container is None:
        return None

    command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="fuel")
    if command is None:
        return None
    return (container, command)


# How long a clearance aim stays suppressed after its shot: the 0x45
# detonation follows the shot echo within a wire tick, so two server
# windows comfortably covers apply-lag without stalling a genuine
# re-clear (a recruit's 1-mine blast leaving covered neighbors).
_MINE_CLEARANCE_EFFECT_MS = 5_000

# Sustained-fire floor for the escape verb law: matches the break
# assessment's own 3-hit window floor so "under fire" means the same
# thing in both places.
_SUSTAINED_FIRE_HIT_FLOOR = 3

# Movement-dead floor: this many server cant_go refusals inside the
# fire window mean the tank cannot walk ANYWHERE right now (boxed by
# terrain, tanks, or unrevealed mines), so the escape skips every
# walk rung and jumps straight to the hop -- a teleport needs no walk
# path and its landing is displacement-safe. Run bot-20260730-110x
# ticks 95-107: twelve consecutive rejected walk-pickups under
# purple-1's fire, fuel 972->663, before the hop rung finally won.
_MOVEMENT_DEAD_REJECTION_FLOOR = 2


def _mine_clearance_decision(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Shoot the best mine-covered container in view, if any.

    User doctrine ([[flag-triage-20260729]] F3: "there is equipment
    that we can see under the orange mines"): a container under an
    enemy mine needs no path clearing -- one clear-line shot at the
    container's tile detonates the covering mine plus the full
    adjacent 3x3 at private+, and the exposed containers become
    ordinary pickups on the very next tick (the 0x45 detonation
    removes them from the world's mine layer). Mine shots consume NO
    inventory (user law 2026-07-30: "shooting a mine doesnt cost any
    inventory. you click and it shoots a single shot") -- the server
    routes a free single -- so the only spend is the shot's tick. The
    aim comes from the pure planner in
    :mod:`tankpit_bot.bot.ai.mine_clearance`; mines never occlude the
    shot line, only rock and movable land blocks do.

    Args:
        ctx: Decision context.
        base_state: AI state for the produced command.

    Returns:
        Shoot decision at the covered container, or ``None`` when no
        covered container in view has a clear line.
    """
    aim = find_mine_clearance_shot(ctx.filtered, ctx.self_state, ctx.terrain)
    if aim is None:
        return None
    aim_x, aim_y = aim
    aim_key = f"{aim_x},{aim_y}"
    if (
        aim_key == base_state["mine_clearance_aim_key"]
        and ctx.timestamp_ms - base_state["mine_clearance_shot_ms"] < _MINE_CLEARANCE_EFFECT_MS
    ):
        # The previous clearance shot at this exact tile has not had
        # its detonation applied yet (the 0x45 follows the shot echo
        # by up to a wire tick); re-aiming inside the window is the
        # live double-shot at (162,94), 01:59:57/:59 -- one shot
        # wasted on mines that were already dead on the server.
        return None
    emit_ai(
        "mine clearance: shooting covered container at (%d,%d) to expose the pickups",
        aim_x,
        aim_y,
    )
    return make_decision(
        make_shoot_command(aim_x, aim_y),
        "COLLECT",
        _COLLECT_SCORE,
        aim_x,
        aim_y,
        "mine_clearance_shot",
        AIStateDict(
            **{
                **base_state,
                "mine_clearance_aim_key": aim_key,
                "mine_clearance_shot_ms": ctx.timestamp_ms,
            }
        ),
        ctx.equip,
    )


# An escape hop must actually ESCAPE: a landing inside the attacker's
# viewport reach keeps the tank in the firing line (flag 1 of run
# bot-20260730-025x: the escape teleported ONE tile, then three, both
# map-open ticks paid, both landings still under red-6's guns --
# because the larder score min(vol, deficit)/cost structurally favors
# the NEAREST fuel, i.e. staying in the kill zone). One full viewport
# of separation is the user-confirmed pursuit horizon: enemies do not
# quickly follow a tank that leaves their view.
_ESCAPE_CLEARANCE_TILES = 16


def _hop_escapes_attacker(
    base_state: AIStateDict,
    decision: TickDecisionDict,
) -> bool:
    """Return True when a hop decision leaves the attacker's reach.

    Args:
        base_state: AI state carrying the held combat lock (the
            attacker the escape is fleeing).
        decision: Candidate hop decision.

    Returns:
        True when there is no known attacker, the decision is not a
        teleport, or the landing clears the attacker's viewport
        envelope.
    """
    if base_state["combat_target_id"] == -1:
        return True
    command = decision["command"]
    if command["cmd_type"] != "teleport":
        return True
    separation = abs(command["target_x"] - base_state["combat_target_x"]) + abs(
        command["target_y"] - base_state["combat_target_y"]
    )
    return separation >= _ESCAPE_CLEARANCE_TILES


def _escape_under_fire_decision(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Escape by hop when collecting under measured sustained fire.

    Escape verb law (Yuppler receipt, run bot-20260730-023x
    02:39:15-21: the break's first escape action was a WALKING fuel
    pickup, fuel bled 640->492 while the attacker landed 4-5 free
    duals -- "he was deciding or teleporting or something"): under
    sustained fire the walk rungs are skipped entirely. Walking keeps
    the tank in the firing line for the whole trip; a hop breaks the
    firing geometry in one action and its landing auto-pickup still
    refuels.

    Args:
        ctx: Decision context.
        base_state: AI state for the produced command.

    Returns:
        Hop (or exhausted-outcome) decision while under fire, or
        ``None`` when no sustained fire is measured and the normal
        cascade should run.
    """
    fire_hits, _fire_fuel = get_incoming_damage_window(ctx.timestamp_ms, INCOMING_RATE_WINDOW_MS)
    if fire_hits < _SUSTAINED_FIRE_HIT_FLOOR:
        return None
    # Movement law under fire (user, 2026-07-30, flag 4 of run
    # bot-20260730-025x): "a tele is 2 ticks. walking is 1 tick. and
    # even if its a long walk you only take one hit. whereas a
    # teleport you can take two hits during." Same-viewport fuel is
    # therefore WALKED -- one action, at most one hit -- and a
    # teleport is only worth its two-hit window when it actually
    # leaves the attacker's envelope.
    emit_ai(
        "collecting under fire (%d hits in window) - walk in-viewport fuel or hop OUT",
        fire_hits,
    )
    # Committed-plan continuity ([[committed-intent]], s8-2 receipt of
    # run bot-20260730-025337: an escape hop landed ON its target and
    # the next derivation re-selected a teleport to the tile the tank
    # was standing on, burning a map-open tick): a held plan whose
    # purpose is served from HERE is finished first — the continuation
    # is one action (a pickup, or the single blessed-under-fire step),
    # so re-deriving can only add exposure, never remove it.
    plan = current_collect_plan(base_state)
    if plan is not None and plan_completes_here(plan, ctx.self_state["x"], ctx.self_state["y"]):
        locked_decision, base_state = _continue_or_release_lock(ctx, base_state)
        if locked_decision is not None:
            return locked_decision
    # Movement-dead check: when the server has refused this tank's
    # movement _MOVEMENT_DEAD_REJECTION_FLOOR times inside the fire
    # window, every further walk plan is fantasy — the walk rung is
    # skipped and the hop (which needs no walk path and lands
    # displacement-safe) is the only escape verb left.
    movement_dead = (
        recent_movement_rejections(ctx.timestamp_ms, INCOMING_RATE_WINDOW_MS)
        >= _MOVEMENT_DEAD_REJECTION_FLOOR
    )
    if movement_dead:
        emit_ai(
            "movement rejected %d+ times in window - walk rungs dead, hopping OUT",
            _MOVEMENT_DEAD_REJECTION_FLOOR,
        )
    if not movement_dead:
        fuel_walk = _select_and_pickup_fuel(ctx, base_state)
        if fuel_walk is not None:
            return fuel_walk
    larder_under_fire = _larder_harvest(ctx, base_state)
    if larder_under_fire is not None and _hop_escapes_attacker(base_state, larder_under_fire):
        return larder_under_fire
    escape_hop = make_resource_search_hop(
        ctx,
        mode="COLLECT",
        score=_COLLECT_SCORE,
        reason="search_collect_local",
        ai_state=base_state,
    )
    if escape_hop is not None and _hop_escapes_attacker(base_state, escape_hop):
        return escape_hop
    # Nothing clears the attacker's envelope: any movement still beats
    # standing in the firing line drinking dregs.
    trapped_fallback = larder_under_fire if larder_under_fire is not None else escape_hop
    if trapped_fallback is not None:
        return trapped_fallback
    return _exhausted_collect_outcome(ctx, base_state)


def decide_collect_mode(ctx: DecideCtx) -> TickDecisionDict | None:
    """Run the durable ``COLLECT`` owner for this tick.

    Cascade:

    1. Continue a held equipment or fuel lock from a previous tick.
    2. Scan-on-landing: fire one radar when the current viewport has
       zero scan coverage. Mirrors HUNT's scan_on_landing so the
       planner has a full picture (0x5A patch entries plus any tiles
       radar reveals) before committing to a pickup order. Without
       this gate, the cascade picks up whatever 0x5A enumerated first
       and only later discovers (via the forage step below) extra
       containers radar would have shown up front.
    3. Pick up the best equipment in the current viewport.
    4. Pick up the best fuel in the current viewport (skipped at cap).
    5. Larder ([[larder-plan]], 2026-07-27): harvest KNOWN stock
       before any discovery -- teleport to tracked equipment when
       below combat-ready, else to the best-scoring tracked fuel
       container (``min(volume, deficit) / cost``, profitable hops
       only). Larder hops hold a resource lock on the target and
       never spend the landing radar.
    6. Forage: radar when the viewport has unscanned tiles, or walk
       toward an unscanned tile so the next free radar covers it.
    7. Hop: teleport to the best-value fuel dot when nothing
       actionable remains here -- candidates are RANKED by
       ``dots_in_landing_viewport * walkable_fraction / cost``, hard
       gates are physics only (landing passable, affordable, not
       freshly scanned; the 2026-07-03 100%-walkable hard filter was
       replaced 2026-07-18 -- it starved the cascade). Landing
       auto-pickup makes the hop partially self-funding. With an
       empty dot atlas the hop opens the map first.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned collection decision, or ``None`` when every cascade
        branch declines but fuel is above ``fuel_low_threshold`` -- the
        tank is stocked, so collection is DONE and the caller should
        hand the tick to the hunt owner. Live run 2026-07-06 hit this
        state at fuel 1100 (every dot-hop landing filtered out) and
        wrongly exited ``out_of_fuel`` instead of going hunting.

    Raises:
        SessionExitError: When every cascade branch declines AND fuel is
            at or below ``fuel_low_threshold`` -- the bot is marooned
            and cannot produce a legal collection action, so the
            session ends with ``out_of_fuel`` (user contract
            2026-07-02).
    """
    base_state = ctx.base
    # Landing scan gates BEFORE lock continuation (reordered 2026-07-30,
    # flag s4-3): the user policy is "always radar right on landing,
    # before any pickup" (2026-07-03), and a DISPLACED harvest landing
    # keeps its lock — running the lock first walked blind into the
    # unobserved minefield three ticks straight. Clean suppressed
    # landings still latch silently here and fall through to the lock.
    landing_scan, base_state = _scan_on_landing_decision(ctx, base_state)
    if landing_scan is not None:
        return landing_scan

    under_fire = _escape_under_fire_decision(ctx, base_state)
    if under_fire is not None:
        return under_fire

    locked_decision, base_state = _continue_or_release_lock(ctx, base_state)
    if locked_decision is not None:
        return locked_decision

    equip_decision = _select_and_pickup_equipment(ctx, base_state)
    if equip_decision is not None:
        return equip_decision

    fuel_decision = _select_and_pickup_fuel(ctx, base_state)
    if fuel_decision is not None:
        return fuel_decision

    emit_ai(
        "no actionable collect target (equipment: %s; fuel: %s)",
        describe_container_search(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            want_fuel=False,
        ),
        describe_container_search(
            ctx.filtered,
            ctx.self_state,
            ctx.terrain,
            want_fuel=True,
            minimum_volume=1,
        ),
    )

    clearance_decision = _mine_clearance_decision(ctx, base_state)
    if clearance_decision is not None:
        return clearance_decision

    larder_decision = _larder_harvest(ctx, base_state)
    if larder_decision is not None:
        return larder_decision

    forage_decision = plan_forage_search(
        ctx,
        base_state,
        score=_COLLECT_SCORE,
        behavior_mode="COLLECT",
        radar_affordable=can_use_radar(ctx),
    )
    if forage_decision is not None:
        return forage_decision

    search = make_resource_search_hop(
        ctx,
        mode="COLLECT",
        score=_COLLECT_SCORE,
        reason="search_collect_local",
        ai_state=base_state,
    )
    if search is not None:
        return search

    return _exhausted_collect_outcome(ctx, base_state)


def _exhausted_collect_outcome(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Resolve a tick where every collect cascade step declined.

    Healthy fuel yields to hunt (or exits ``no_productive_collect``
    when under-stocked); critical fuel gets the walk-for-fuel last
    resort before the ``out_of_fuel`` exit.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for a walk decision.

    Returns:
        ``None`` to hand the tick to hunt, or a walk decision.

    Raises:
        SessionExitError: When the session has no productive action.
    """
    if ctx.fuel > ctx.config["fuel_low_threshold"]:
        if hunt_entry_permitted(ctx):
            emit_ai(
                "collect exhausted at (%d,%d) fuel=%d combat-ready, yielding to hunt",
                ctx.self_state["x"],
                ctx.self_state["y"],
                ctx.fuel,
            )
            return None
        raise SessionExitError(
            "no_productive_collect",
            f"COLLECT owner produced no decision at "
            f"({ctx.self_state['x']},{ctx.self_state['y']}) fuel={ctx.fuel} "
            f"dual={ctx.inventory['dual_shots']['count']} "
            f"homing={ctx.inventory['homing_shots']['count']} "
            f"radar={ctx.inventory['extra_radars']['count']}: "
            f"inventory below combat-ready and no reachable equipment.",
        )

    desperation = _desperation_fuel_hop(ctx, base_state)
    if desperation is not None:
        return desperation

    walk = _walk_for_fuel_last_resort(ctx, base_state)
    if walk is not None:
        return walk

    raise SessionExitError(
        "out_of_fuel",
        f"COLLECT owner produced no decision at "
        f"({ctx.self_state['x']},{ctx.self_state['y']}) fuel={ctx.fuel}: "
        f"forager exhausted, no affordable search hop, no walkable fuel "
        f"within {_WALK_FOR_FUEL_MAX_TILES} tiles.",
    )


def _desperation_fuel_hop(
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
    physics only: a legal landing (`find_teleport_landing_tile`, the
    shore-aware helper) and cost within the remaining tank. The hop
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
        landing = find_teleport_landing_tile(terrain, container["x"], container["y"])
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
        _COLLECT_SCORE,
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


_WALK_FOR_FUEL_MAX_TILES = 48
"""Farthest known fuel a marooned tank will walk toward (~96 s of
2 s-per-tile walking). Beyond this the out_of_fuel exit stands -- the
2026-07-25 exposure rule caps how long a broke tank crawls in the
open, even in the practice room where bots never initiate."""


def _walk_for_fuel_last_resort(
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
        c for c in candidates if 0 < c[0] <= _WALK_FOR_FUEL_MAX_TILES
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
            _COLLECT_SCORE,
            leg_x,
            leg_y,
            "walk_for_fuel",
            release_collect_plan(base_state, reason="walk_for_fuel_override"),
            ctx.equip,
        )
    return None


def _scan_on_landing_decision(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    """Return a radar decision when this viewport has no landing scan yet.

    The COLLECT-mode equivalent of HUNT's ``scan_on_landing``: fired
    once per viewport entry, before any pickup logic runs. The gate is
    the ``last_landing_scan_viewport`` latch -- the viewport origin
    changes only on teleport, so "origin differs from the latch" means
    the bot landed here without radaring yet. User policy (2026-07-03):
    always radar right on landing, unconditionally — the 0x5A patch is
    truthful for the visible layer but says nothing about hidden
    containers, and re-entering previously scanned ground is exactly
    when coverage marks are most stale. (The previous zero-coverage
    gate skipped the scan whenever the 18-wide visible viewport
    overlapped 2 tiles of old coverage after a 16-tile hop.)

    Larder exception (user ruling 2026-07-27, [[larder-plan]]): a
    landing flagged ``suppress_landing_scan`` is a harvest hop to
    already-verified stock -- the latch records the viewport WITHOUT
    dispatching the radar and the flag is consumed, so the cascade
    proceeds straight to the pickup this tick.

    Displacement exception to the exception (flag s4-3,
    [[flag-triage-20260729]]): a harvest hop expects to stand within
    auto-pick reach of its locked target. Standing farther means the
    server displaced the landing — unobserved mines shoved it — or the
    landing was a ferry boarding; either way the ground is NOT the
    verified stock the no-radar ruling assumed, and walking blind ate
    three straight ``cant_go`` rejections at 01:28. The radar fires,
    the LOCK IS KEPT (the target is still valid; the mine-composed
    passability decides the re-approach), and the suppression is
    consumed.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        ``(decision, base_state)`` -- the ``scan_on_landing`` decision
        (or ``None`` when this viewport already had its landing radar,
        or the larder flag consumed it) and the state the remaining
        cascade must thread.
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    origin_key = f"{left},{top}"
    if base_state["last_landing_scan_viewport"] == origin_key:
        return None, base_state
    if base_state["suppress_landing_scan"]:
        lock_dist = abs(ctx.self_state["x"] - base_state["resource_target_x"]) + abs(
            ctx.self_state["y"] - base_state["resource_target_y"]
        )
        if base_state["resource_target_kind"] == "" or lock_dist <= 1:
            emit_ai(
                "larder landing at viewport (%d,%d)-(%d,%d): latching without radar",
                left,
                top,
                right,
                bottom,
            )
            return None, AIStateDict(
                **{
                    **base_state,
                    "last_landing_scan_viewport": origin_key,
                    "suppress_landing_scan": False,
                }
            )
        emit_ai(
            "harvest landing displaced: self (%d,%d) is %d tiles from lock (%d,%d)"
            " - un-suppressing landing radar",
            ctx.self_state["x"],
            ctx.self_state["y"],
            lock_dist,
            base_state["resource_target_x"],
            base_state["resource_target_y"],
        )
        displaced_scan = make_decision(
            make_radar_command(),
            "COLLECT",
            _COLLECT_SCORE,
            0,
            0,
            "scan_on_landing",
            AIStateDict(
                **{
                    **base_state,
                    "last_landing_scan_viewport": origin_key,
                    "suppress_landing_scan": False,
                }
            ),
            ctx.equip,
        )
        return displaced_scan, base_state
    emit_ai(
        "scan-on-landing (mode=COLLECT, extras=%d, viewport=(%d,%d)-(%d,%d))",
        ctx.inventory["extra_radars"]["count"],
        left,
        top,
        right,
        bottom,
    )
    decision = make_decision(
        make_radar_command(),
        "COLLECT",
        _COLLECT_SCORE,
        0,
        0,
        "scan_on_landing",
        AIStateDict(
            **{
                **release_collect_plan(base_state, reason="landing_scan_reset"),
                "last_landing_scan_viewport": origin_key,
            }
        ),
        ctx.equip,
    )
    return decision, base_state


def _continue_or_release_lock(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    """Resolve any held resource lock for this tick.

    Looks up the lock kind from ``ai_state.resource_target_kind`` and
    routes to the matching continue/release path. A cleared lock leaves
    ``base_state`` with ``resource_target_kind == ""`` so downstream
    cascade steps see a clean slate.
    """
    _, equipment_lock = locked_resource_target(ctx, "equipment")
    if equipment_lock is not None:
        return _continue_or_release_equipment_lock(ctx, base_state, equipment_lock)
    _, fuel_lock = locked_resource_target(ctx, "fuel")
    if fuel_lock is not None:
        return _continue_or_release_fuel_lock(ctx, base_state, fuel_lock)
    return None, base_state


def _continue_or_release_equipment_lock(
    ctx: DecideCtx,
    base_state: AIStateDict,
    locked_target: ContainerStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    if _superior_equipment_candidate(ctx, locked_target) is not None:
        emit_ai(
            "releasing equipment lock at (%d,%d): markedly closer equipment is visible",
            locked_target["x"],
            locked_target["y"],
        )
        return None, release_collect_plan(base_state, reason="superior_candidate")
    target_x = locked_target["x"]
    target_y = locked_target["y"]
    locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")
    if locked_command is None:
        # Transient inexecutability HOLDS the plan ([[committed-intent]];
        # run bot-20260730-032x ticks 361/366/371: three not_executable
        # releases fired mid-approach with the plan's own map_open in
        # flight, and each target was re-locked and served 2-3 ticks
        # later — the plan was never invalid, the executor was busy).
        # Only the server-confirmed move-failed mark is structural.
        if is_move_target_failed(target_x, target_y, ctx.timestamp_ms):
            emit_ai(
                "locked equipment target at (%d,%d) marked move-failed - releasing",
                target_x,
                target_y,
            )
            return None, release_collect_plan(base_state, reason="not_executable")
        emit_ai(
            "locked equipment target at (%d,%d) not executable this tick - holding plan",
            target_x,
            target_y,
        )
        return None, base_state
    emit_ai("continue locked equipment target at (%d,%d)", target_x, target_y)
    decision = make_decision(
        locked_command,
        "COLLECT",
        _COLLECT_SCORE,
        target_x,
        target_y,
        "equipment_locked",
        set_resource_target(base_state, "equipment", target_x, target_y),
        ctx.equip,
    )
    return decision, base_state


def _continue_or_release_fuel_lock(
    ctx: DecideCtx,
    base_state: AIStateDict,
    locked_target: ContainerStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    if ctx.fuel >= fuel_capacity(ctx.self_state["rank"]):
        # Live run 2026-07-06 pickup loop: a held fuel lock kept
        # re-dispatching pickup_fuel at capacity because the lock path
        # had no capacity gate, only the selection path did. Every
        # dispatch drew wire 0x52 code-5 "Tank full" and the lock
        # survived to next tick. Capacity is now rank-derived
        # (:func:`tankpit_bot.physics.capacity.fuel_capacity`), so
        # this gate closes the loop at the root regardless of how the
        # lock was established.
        emit_ai(
            "releasing fuel lock at (%d,%d): tank at capacity %d",
            locked_target["x"],
            locked_target["y"],
            ctx.fuel,
        )
        return None, release_collect_plan(base_state, reason="tank_at_capacity")
    if _superior_fuel_candidate(ctx, locked_target) is not None:
        emit_ai(
            "releasing fuel lock at (%d,%d): markedly closer fuel is visible",
            locked_target["x"],
            locked_target["y"],
        )
        return None, release_collect_plan(base_state, reason="superior_candidate")
    target_x = locked_target["x"]
    target_y = locked_target["y"]
    locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="fuel")
    if locked_command is None:
        # Same transient-vs-structural law as the equipment lock.
        if is_move_target_failed(target_x, target_y, ctx.timestamp_ms):
            emit_ai(
                "locked fuel target at (%d,%d) marked move-failed - releasing",
                target_x,
                target_y,
            )
            return None, release_collect_plan(base_state, reason="not_executable")
        emit_ai(
            "locked fuel target at (%d,%d) not executable this tick - holding plan",
            target_x,
            target_y,
        )
        return None, base_state
    emit_ai(
        "continue locked fuel target at (%d,%d) vol=%d (fuel=%d)",
        target_x,
        target_y,
        locked_target["volume"],
        ctx.fuel,
    )
    decision = make_decision(
        locked_command,
        "COLLECT",
        _COLLECT_SCORE,
        target_x,
        target_y,
        "fuel_locked",
        set_resource_target(base_state, "fuel", target_x, target_y),
        ctx.equip,
        reason_context={"volume": locked_target["volume"]},
    )
    return decision, base_state


def _select_and_pickup_equipment(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Pick up the best viewport equipment, unless every slot is full.

    User mechanic (2026-07-18): containers fill whatever is empty and
    the server rejects with code 7 only at all-slots-full -- so at
    full inventory a pickup can gain nothing and would burn a tick on
    a guaranteed rejection (8 wasted ticks in the 2026-07-18 5-minute
    run before this gate).
    """
    if inventory_all_full(ctx.inventory, inventory_capacity(ctx.self_state["rank"])):
        return None
    selection = select_equipment_target(ctx)
    if selection is None:
        return None
    container, command = selection
    target_x = container["x"]
    target_y = container["y"]
    emit_ai("collect equipment at (%d,%d)", target_x, target_y)
    return make_decision(
        command,
        "COLLECT",
        _COLLECT_SCORE,
        target_x,
        target_y,
        "equipment_restock",
        set_resource_target(base_state, "equipment", target_x, target_y),
        ctx.equip,
    )


def _emit_hop_declined(hop_kind: str, **tallies: int) -> None:
    """Record a structured hop decline with per-branch tallies.

    The hop selectors' silent ``continue``/``return None`` branches
    made the 2026-07-18 early-exit undiagnosable post-hoc (the run
    ended ``no_productive_collect`` with 10 tracked containers and no
    record of which filter refused each). Every decline now states
    its arithmetic.

    Args:
        hop_kind: Which selector declined (``equipment`` / ``dot``).
        **tallies: Per-branch counts and the governing numbers.
    """
    emit_diagnostic(diagnostic_kind="hop_declined", hop_kind=hop_kind, **tallies)


def _hop_toward_equipment(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Teleport toward tracked equipment the walk-pickup step cannot reach.

    ``find_nearest_equipment`` filters to walkable-from-here candidates
    inside the current viewport; step 3 of the cascade
    (``_select_and_pickup_equipment``) handles those and runs FIRST.
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
        combat-ready, terrain is unknown, nothing is tracked, no
        candidate has a legal landing tile, or every affordable
        teleport would leave the engagement reserve.
    """
    if hunt_entry_permitted(ctx):
        return None
    if ctx.terrain is None:
        return None
    candidates = find_all_tracked_equipment(ctx.world)
    if not candidates:
        _emit_hop_declined("equipment", no_candidates=1)
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
        landing = find_teleport_landing_tile(
            ctx.terrain,
            container["x"],
            container["y"],
        )
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
        _emit_hop_declined(
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
        _COLLECT_SCORE,
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


def _larder_harvest(
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
    equipment_hop = _hop_toward_equipment(ctx, base_state)
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
            _emit_hop_declined(
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
        _COLLECT_SCORE,
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


_FUEL_GAIN_PER_WALK_TILE = 25
"""Minimum effective fuel gained per tile of walking for a pickup to pay.

Each walk tile costs roughly one 2-second tick the bot could spend on
a dot hop, which refuels while traveling. 25/tile keeps adjacent
pickups always worth taking (walk 0-1 clears at any meaningful gain)
while refusing long walks for cap-clamped slivers -- the 2026-07-06
waste class (26 s across four near-cap pickups for ~50 fuel each).
"""


def _pickup_not_worth_walk(
    ctx: DecideCtx,
    container: ContainerStateDict,
) -> bool:
    """Return True when the pickup's real transfer is not worth the walk.

    The server clamps a fuel pickup to ``min(volume, headroom)`` and
    answers ``code=5`` when clamped; since 2026-07-06/-19 that code is
    handled cleanly (container kept, no blacklist, ledger resolved),
    so a clamped transfer costs nothing but the walk. The predicate
    therefore rates the ACTUAL transfer against the walk distance.

    This replaces the binary overfill gate, whose formula
    (``fuel + walk + min(volume, headroom) > cap``) refused ANY
    clamped pickup at walk >= 1 -- and refused earlier the bigger the
    container: at fuel 600 it walked past a 1000-volume container one
    tile away, forfeiting a 500-fuel transfer (falsified 2026-07-19;
    the 2026-06-23 minimum-volume lesson -- fuel is fuel -- applies at
    the cap end too).

    Args:
        ctx: Decision context.
        container: The candidate fuel container.

    Returns:
        True when ``min(volume, headroom)`` falls below
        ``_FUEL_GAIN_PER_WALK_TILE`` per tile of Manhattan walk.
    """
    headroom = fuel_capacity(ctx.self_state["rank"]) - ctx.fuel
    effective_gain = min(container["volume"], headroom)
    walk_tiles = abs(container["x"] - ctx.self_state["x"]) + abs(
        container["y"] - ctx.self_state["y"]
    )
    return effective_gain < _FUEL_GAIN_PER_WALK_TILE * walk_tiles


def _select_and_pickup_fuel(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    if ctx.fuel >= fuel_capacity(ctx.self_state["rank"]):
        return None
    selection = select_fuel_target(ctx)
    if selection is None:
        return None
    container, command = selection
    target_x = container["x"]
    target_y = container["y"]
    # The worth-the-walk rate predicate is an EFFICIENCY rule for a
    # healthy tank; below the fuel-low break the alternative to a
    # "wasteful" walk is the out_of_fuel exit, so any reachable fuel
    # is taken (run bot-20260728-090813: exited at fuel 98 with a
    # pickable 39-fuel container two tiles away, refused as
    # "not worth 2-tile walk").
    fuel_critical = ctx.fuel <= ctx.config["fuel_low_threshold"]
    if not fuel_critical and _pickup_not_worth_walk(ctx, container):
        cap = fuel_capacity(ctx.self_state["rank"])
        walk_tiles = abs(target_x - ctx.self_state["x"]) + abs(target_y - ctx.self_state["y"])
        emit_ai(
            "skip fuel at (%d,%d) vol=%d: clamped gain %d not worth %d-tile walk (fuel=%d cap=%d)",
            target_x,
            target_y,
            container["volume"],
            min(container["volume"], cap - ctx.fuel),
            walk_tiles,
            ctx.fuel,
            cap,
        )
        return None
    emit_ai(
        "collect fuel at (%d,%d) vol=%d (fuel=%d)",
        target_x,
        target_y,
        container["volume"],
        ctx.fuel,
    )
    return make_decision(
        command,
        "COLLECT",
        _COLLECT_SCORE,
        target_x,
        target_y,
        "fuel_collect",
        set_resource_target(base_state, "fuel", target_x, target_y),
        ctx.equip,
        reason_context={"volume": container["volume"]},
    )


def _superior_equipment_candidate(
    ctx: DecideCtx,
    locked_target: ContainerStateDict,
) -> ContainerStateDict | None:
    candidate = find_nearest_equipment(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
    )
    if candidate is None:
        return None
    if (candidate["x"], candidate["y"]) == (locked_target["x"], locked_target["y"]):
        return None
    if not is_lock_release_warranted(
        ctx.self_state,
        locked_target["x"],
        locked_target["y"],
        candidate["x"],
        candidate["y"],
    ):
        return None
    return candidate


def _superior_fuel_candidate(
    ctx: DecideCtx,
    locked_target: ContainerStateDict,
) -> ContainerStateDict | None:
    candidate = find_best_fuel(
        ctx.filtered,
        ctx.self_state,
        ctx.terrain,
        minimum_volume=1,
    )
    if candidate is None:
        return None
    if (candidate["x"], candidate["y"]) == (locked_target["x"], locked_target["y"]):
        return None
    deficit = fuel_capacity(ctx.self_state["rank"]) - ctx.fuel
    if not is_fuel_lock_release_warranted(ctx.self_state, locked_target, candidate, deficit):
        return None
    return candidate


__all__ = [
    "decide_collect_mode",
    "is_container_blacklisted",
    "reset_container_blacklist",
    "select_equipment_target",
    "select_fuel_target",
]
