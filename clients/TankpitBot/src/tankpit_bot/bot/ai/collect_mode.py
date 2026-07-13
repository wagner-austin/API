"""Durable COLLECT owner: unified fuel-and-equipment recovery cascade.

The user's gameplay loop is one mode: drain the equipment in the
current viewport, drain the fuel (when below cap), radar/walk to expand
coverage, and teleport to a fresh viewport when nothing actionable
remains here.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_use_radar,
    clear_resource_target,
    locked_resource_target,
    make_decision,
    set_resource_target,
)
from tankpit_bot.bot.ai.equipment import is_lock_release_warranted
from tankpit_bot.bot.ai.equipment_search import (
    describe_container_search,
    find_all_tracked_equipment,
    find_best_fuel,
    find_equipment_candidates,
    find_nearest_equipment,
    find_teleport_landing_tile,
)
from tankpit_bot.bot.ai.forage import plan_forage_search
from tankpit_bot.bot.ai.mode_controller import hunt_entry_permitted
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.resource_search import (
    make_resource_search_hop,
)
from tankpit_bot.bot.ai.teleport_cost import compute_teleport_fuel_cost
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand, make_radar_command, make_teleport_command
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.state.rank_formulas import fuel_capacity
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
    5. Forage: radar when the viewport has unscanned tiles, or walk
       toward an unscanned tile so the next free radar covers it.
    6. Hop: teleport to the nearest fuel dot whose landing viewport
       is unscanned and 100% walkable when nothing actionable remains
       here (landing auto-pickup makes the hop partially
       self-funding). With an empty dot atlas the hop opens the map
       first.

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
    locked_decision, base_state = _continue_or_release_lock(ctx, base_state)
    if locked_decision is not None:
        return locked_decision

    landing_scan = _scan_on_landing_decision(ctx, base_state)
    if landing_scan is not None:
        return landing_scan

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

    forage_decision = plan_forage_search(
        ctx,
        base_state,
        score=_COLLECT_SCORE,
        behavior_mode="COLLECT",
        radar_affordable=can_use_radar(ctx),
    )
    if forage_decision is not None:
        return forage_decision

    equipment_hop = _hop_toward_equipment(ctx, base_state)
    if equipment_hop is not None:
        return equipment_hop

    search = make_resource_search_hop(
        ctx,
        mode="COLLECT",
        score=_COLLECT_SCORE,
        reason="search_collect_local",
        ai_state=base_state,
    )
    if search is not None:
        return search

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

    raise SessionExitError(
        "out_of_fuel",
        f"COLLECT owner produced no decision at "
        f"({ctx.self_state['x']},{ctx.self_state['y']}) fuel={ctx.fuel}: "
        f"forager exhausted, no affordable search hop.",
    )


def _scan_on_landing_decision(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
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

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        ``scan_on_landing`` decision, or ``None`` when this viewport
        already had its landing radar.
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    origin_key = f"{left},{top}"
    if base_state["last_landing_scan_viewport"] == origin_key:
        return None
    emit_ai(
        "scan-on-landing (mode=COLLECT, extras=%d, viewport=(%d,%d)-(%d,%d))",
        ctx.inventory["extra_radars"]["count"],
        left,
        top,
        right,
        bottom,
    )
    return make_decision(
        make_radar_command(),
        "COLLECT",
        _COLLECT_SCORE,
        0,
        0,
        "scan_on_landing",
        AIStateDict(
            **{
                **clear_resource_target(base_state),
                "last_landing_scan_viewport": origin_key,
            }
        ),
        ctx.equip,
    )


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
        return None, clear_resource_target(base_state)
    target_x = locked_target["x"]
    target_y = locked_target["y"]
    locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="equipment")
    if locked_command is None:
        emit_ai("locked equipment target at (%d,%d) no longer executable", target_x, target_y)
        return None, clear_resource_target(base_state)
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
        # (:func:`tankpit_bot.state.rank_formulas.fuel_capacity`), so
        # this gate closes the loop at the root regardless of how the
        # lock was established.
        emit_ai(
            "releasing fuel lock at (%d,%d): tank at capacity %d",
            locked_target["x"],
            locked_target["y"],
            ctx.fuel,
        )
        return None, clear_resource_target(base_state)
    if _superior_fuel_candidate(ctx, locked_target) is not None:
        emit_ai(
            "releasing fuel lock at (%d,%d): markedly closer fuel is visible",
            locked_target["x"],
            locked_target["y"],
        )
        return None, clear_resource_target(base_state)
    target_x = locked_target["x"]
    target_y = locked_target["y"]
    locked_command = walk_or_teleport(ctx, target_x, target_y, pickup_kind="fuel")
    if locked_command is None:
        emit_ai("locked fuel target at (%d,%d) no longer executable", target_x, target_y)
        return None, clear_resource_target(base_state)
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
        f"fuel={locked_target['volume']}",
        set_resource_target(base_state, "fuel", target_x, target_y),
        ctx.equip,
    )
    return decision, base_state


def _select_and_pickup_equipment(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
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


def _hop_toward_equipment(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    """Teleport toward tracked equipment outside the current viewport.

    ``find_nearest_equipment`` filters to walkable-from-here candidates
    inside the current viewport; step 3 of the cascade
    (``_select_and_pickup_equipment``) handles those. Everything else
    ``world.containers`` has ever accumulated -- equipment revealed by
    prior radar or 0x5A patches in other viewports -- is invisible to
    the viewport-scoped picks and was silently ignored until Bug 0.7.

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
        Teleport decision to the closest reachable + affordable
        out-of-viewport equipment container, or ``None`` when
        inventory is combat-ready, terrain is unknown, no tracked
        equipment sits outside the viewport, no candidate has a
        legal landing tile, or every affordable teleport would
        leave the engagement reserve.
    """
    if hunt_entry_permitted(ctx):
        return None
    if ctx.terrain is None:
        return None
    candidates = find_all_tracked_equipment(ctx.world)
    if not candidates:
        return None
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    external = [
        container
        for container in candidates
        if not (left <= container["x"] <= right and top <= container["y"] <= bottom)
    ]
    if not external:
        return None
    landing_reserve = ctx.config["engagement_fuel_budget"] + ctx.config["fuel_low_threshold"]
    best_cost = -1
    best_landing_x = 0
    best_landing_y = 0
    best_container: ContainerStateDict | None = None
    for container in external:
        landing = find_teleport_landing_tile(
            ctx.terrain,
            ctx.self_state["x"],
            ctx.self_state["y"],
            container["x"],
            container["y"],
        )
        if landing is None:
            continue
        landing_x, landing_y = landing
        cost = compute_teleport_fuel_cost(
            ctx.self_state["x"],
            ctx.self_state["y"],
            landing_x,
            landing_y,
        )
        if ctx.fuel - cost < landing_reserve:
            continue
        if best_container is None or cost < best_cost:
            best_cost = cost
            best_landing_x = landing_x
            best_landing_y = landing_y
            best_container = container
    if best_container is None:
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
    return make_decision(
        make_teleport_command(best_landing_x, best_landing_y),
        "COLLECT",
        _COLLECT_SCORE,
        best_landing_x,
        best_landing_y,
        "equipment_hop",
        clear_resource_target(base_state),
        ctx.equip,
    )


def _would_overfill(
    ctx: DecideCtx,
    container: ContainerStateDict,
) -> bool:
    """Return True when picking up ``container`` would exceed fuel cap.

    A pickup that transfers less than the container's full volume
    (because the tank is near cap) draws server ``code=5`` and marks
    the container ``failed_pickup``; the 2026-07-06 22:37 run spent
    26 s dispatching four consecutive overflow pickups at fuel
    1040/1054/1062/1054 (headroom 46/60/46) before every nearby fuel
    container was blacklisted. Refuse the dispatch instead of paying
    the walk cost and blacklisting the container.

    Formula matches the handoff spec: the projected end-state of the
    walk plus the server's clamped transfer (``min(volume, headroom)``)
    exceeds cap. When ``walk_cost > 0`` any container whose volume
    meets or exceeds current headroom will trigger a refusal, since
    the transfer would fill to cap and still leave fuel in the
    container -- exactly the wasteful-pickup class the server rejects.

    Args:
        ctx: Decision context.
        container: The candidate fuel container.

    Returns:
        True when the projected pickup would exceed
        :func:`fuel_capacity` for the current rank.
    """
    cap = fuel_capacity(ctx.self_state["rank"])
    walk_cost = abs(container["x"] - ctx.self_state["x"]) + abs(
        container["y"] - ctx.self_state["y"]
    )
    headroom = cap - ctx.fuel
    projected = ctx.fuel + walk_cost + min(container["volume"], headroom)
    return projected > cap


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
    if _would_overfill(ctx, container):
        cap = fuel_capacity(ctx.self_state["rank"])
        walk_cost = abs(target_x - ctx.self_state["x"]) + abs(target_y - ctx.self_state["y"])
        emit_ai(
            "skip fuel at (%d,%d) vol=%d: would overfill (fuel=%d cap=%d walk=%d headroom=%d)",
            target_x,
            target_y,
            container["volume"],
            ctx.fuel,
            cap,
            walk_cost,
            cap - ctx.fuel,
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
        f"fuel={container['volume']}",
        set_resource_target(base_state, "fuel", target_x, target_y),
        ctx.equip,
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
    if not is_lock_release_warranted(
        ctx.self_state,
        locked_target["x"],
        locked_target["y"],
        candidate["x"],
        candidate["y"],
    ):
        return None
    return candidate


__all__ = [
    "decide_collect_mode",
    "is_container_blacklisted",
    "reset_container_blacklist",
    "select_equipment_target",
    "select_fuel_target",
]
