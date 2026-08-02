"""Durable COLLECT owner: unified fuel-and-equipment recovery cascade.

The user's gameplay loop is one mode: drain the equipment in the
current viewport, drain the fuel (when below cap), radar/walk to expand
coverage, and teleport to a fresh viewport when nothing actionable
remains here. This module owns the cascade order and the sense/safety
gates; pickups live in :mod:`collect_pickups`, lock continuation in
:mod:`collect_locks`, harvest hops and the marooned ladder in
:mod:`collect_hops`, shared primitives in :mod:`collect_common`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_common import COLLECT_SCORE
from tankpit_bot.bot.ai.collect_hops import (
    WALK_FOR_FUEL_MAX_TILES,
    desperation_fuel_hop,
    larder_harvest,
    walk_for_fuel_last_resort,
)
from tankpit_bot.bot.ai.collect_locks import continue_or_release_lock
from tankpit_bot.bot.ai.collect_pickups import (
    mine_clearance_decision,
    select_and_pickup_equipment,
    select_and_pickup_fuel,
)
from tankpit_bot.bot.ai.combat_break import INCOMING_RATE_WINDOW_MS
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    make_decision,
    radar_spend_worthwhile,
)
from tankpit_bot.bot.ai.equipment_search import describe_container_search
from tankpit_bot.bot.ai.forage import plan_forage_search
from tankpit_bot.bot.ai.intent import (
    current_collect_plan,
    plan_completes_here,
    release_collect_plan,
)
from tankpit_bot.bot.ai.mode_controller import hunt_entry_permitted
from tankpit_bot.bot.ai.resource_search import (
    make_resource_search_hop,
)
from tankpit_bot.bot.ai.scope_scout import scope_scout_for_ferry
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_radar_command
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.sniffer.world_state import (
    clear_container_desync,
    container_desync_pending,
    get_incoming_damage_window,
    recent_movement_rejections,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

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
        locked_decision, base_state = continue_or_release_lock(ctx, base_state)
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
        fuel_walk = select_and_pickup_fuel(ctx, base_state)
        if fuel_walk is not None:
            return fuel_walk
    larder_under_fire = larder_harvest(ctx, base_state)
    if larder_under_fire is not None and _hop_escapes_attacker(base_state, larder_under_fire):
        return larder_under_fire
    escape_hop = make_resource_search_hop(
        ctx,
        mode="COLLECT",
        score=COLLECT_SCORE,
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

    0. Desync rescan: while a code=4 disproof of a remembered
       container awaits its radar resync, one radar outranks every
       pursuit branch (user ruling 2026-07-30: one stale item is
       worth a radar, never a second blind hop).
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
    gate_decision, base_state = _sense_and_safety_gates(ctx, ctx.base)
    if gate_decision is not None:
        return gate_decision

    locked_decision, base_state = continue_or_release_lock(ctx, base_state)
    if locked_decision is not None:
        return locked_decision

    equip_decision = select_and_pickup_equipment(ctx, base_state)
    if equip_decision is not None:
        return equip_decision

    fuel_decision = select_and_pickup_fuel(ctx, base_state)
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

    clearance_decision = mine_clearance_decision(ctx, base_state)
    if clearance_decision is not None:
        return clearance_decision

    larder_decision = larder_harvest(ctx, base_state)
    if larder_decision is not None:
        return larder_decision

    # The larder just declined — if what stopped it was a water-locked
    # container with no fresh ferry belief, one FREE viewport pan at
    # the water beats paying for discovery ([[viewport-shift-protocol]]
    # scope scout; F5 ferry doctrine). A ferry the pan reveals makes
    # the next tick's larder hop ``ferry_served``.
    scout_decision = scope_scout_for_ferry(ctx, base_state)
    if scout_decision is not None:
        return scout_decision

    forage_decision = plan_forage_search(
        ctx,
        base_state,
        score=COLLECT_SCORE,
        behavior_mode="COLLECT",
        radar_affordable=can_use_radar(ctx),
    )
    if forage_decision is not None:
        return forage_decision

    search = make_resource_search_hop(
        ctx,
        mode="COLLECT",
        score=COLLECT_SCORE,
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

    desperation = desperation_fuel_hop(ctx, base_state)
    if desperation is not None:
        return desperation

    walk = walk_for_fuel_last_resort(ctx, base_state)
    if walk is not None:
        return walk

    raise SessionExitError(
        "out_of_fuel",
        f"COLLECT owner produced no decision at "
        f"({ctx.self_state['x']},{ctx.self_state['y']}) fuel={ctx.fuel}: "
        f"forager exhausted, no affordable search hop, no walkable fuel "
        f"within {WALK_FOR_FUEL_MAX_TILES} tiles.",
    )


def _sense_and_safety_gates(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> tuple[TickDecisionDict | None, AIStateDict]:
    """Run the pre-pursuit gates: landing scan, escape, desync rescan.

    Landing scan gates BEFORE lock continuation (reordered 2026-07-30,
    flag s4-3): the user policy is "always radar right on landing,
    before any pickup" (2026-07-03), and a DISPLACED harvest landing
    keeps its lock -- running the lock first walked blind into the
    unobserved minefield three ticks straight. Clean suppressed
    landings still latch silently and fall through to the lock. The
    under-fire escape and the desync rescan follow in that order:
    survival beats resync, resync beats pursuit.

    Args:
        ctx: Decision context.
        base_state: Base AI state threaded through the gates.

    Returns:
        ``(decision, base_state)`` -- the first gate's decision (or
        ``None`` when all gates pass) and the state the remaining
        cascade must thread.
    """
    landing_scan, base_state = _scan_on_landing_decision(ctx, base_state)
    if landing_scan is not None:
        return landing_scan, base_state

    under_fire = _escape_under_fire_decision(ctx, base_state)
    if under_fire is not None:
        return under_fire, base_state

    return _desync_rescan_decision(ctx, base_state), base_state


def _desync_rescan_decision(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Return a radar decision while a container desync awaits resync.

    A code=4 empty-container rejection disproves the belief the
    planner acted on, and the user ruling (2026-07-30) is that ONE
    disproof is worth a radar -- never a second blind hop to another
    remembered container. Session 4 receipt: three larder hops in a
    row landed on containers Yuppler had already collected, each
    landing scan suppressed as verified stock, three teleports wasted.
    The latch (``mark_container_desync``) is set by the rejection
    consumer and cleared by the radar response itself, which
    reconciles the viewport authoritatively (volume==0 entries are
    removals) -- so this gate fires exactly once per disproof.

    Args:
        ctx: Decision context.
        base_state: Base AI state for the produced command.

    Returns:
        The ``desync_rescan`` radar decision, or ``None`` when no
        disproof is pending.
    """
    if not container_desync_pending():
        return None
    if not radar_spend_worthwhile(ctx):
        # Live coverage already tells the whole story (radar-spend
        # economics, flag s9-4: two rescans of ground radared seconds
        # earlier) -- the disproof is answered by the existing scan.
        emit_ai("container desync answered by live coverage - no rescan needed")
        clear_container_desync()
        return None
    emit_ai(
        "remembered container disproved (code=4) - radar resync before "
        "pursuing further memory (extras=%d)",
        ctx.inventory["extra_radars"]["count"],
    )
    return make_decision(
        make_radar_command(),
        "COLLECT",
        COLLECT_SCORE,
        0,
        0,
        "desync_rescan",
        base_state,
        ctx.equip,
    )


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
        if not radar_spend_worthwhile(ctx):
            # Radar-spend economics (flag s9-2): the displaced landing
            # sits in ground live coverage already explains -- the
            # mines the un-suppression exists to reveal are known, so
            # the scan buys nothing. Latch and proceed.
            emit_ai(
                "harvest landing displaced but viewport coverage is live - "
                "no radar spend, proceeding",
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
            COLLECT_SCORE,
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
    if not radar_spend_worthwhile(ctx):
        # Radar-spend economics (flags s9-2/4/5, superseding the
        # 2026-07-03 "always radar right on landing, unconditionally"
        # ruling): a landing in ground live coverage already explains
        # does not buy an extra radar. The latch still records the
        # viewport so the landing is not re-evaluated every tick.
        emit_ai(
            "landing viewport coverage is live - skipping landing radar (extras=%d)",
            ctx.inventory["extra_radars"]["count"],
        )
        return None, AIStateDict(
            **{
                **base_state,
                "last_landing_scan_viewport": origin_key,
            }
        )
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
        COLLECT_SCORE,
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


__all__ = [
    "decide_collect_mode",
]
