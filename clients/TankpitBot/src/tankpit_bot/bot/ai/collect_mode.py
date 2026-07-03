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
    find_best_fuel,
    find_equipment_candidates,
    find_nearest_equipment,
)
from tankpit_bot.bot.ai.forage import plan_forage_search
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.resource_search import (
    make_resource_search_hop,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand, make_radar_command
from tankpit_bot.diagnostics.game_log_feedback import is_fuel_at_learned_capacity
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.state.scan_coverage import is_tile_covered
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
            now_ms=ctx.timestamp_ms,
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
        now_ms=ctx.timestamp_ms,
        minimum_volume=1,
    )
    if container is None:
        return None

    command = walk_or_teleport(ctx, container["x"], container["y"], pickup_kind="fuel")
    if command is None:
        return None
    return (container, command)


def decide_collect_mode(ctx: DecideCtx) -> TickDecisionDict:
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
    6. Hop: teleport to a fresh viewport when nothing actionable
       remains here.

    Args:
        ctx: Decision context.

    Returns:
        Mode-owned collection decision.

    Raises:
        SessionExitError: When every cascade branch declines -- the bot
            is marooned and cannot produce a legal collection action,
            so the session ends with ``out_of_fuel`` (user contract
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

    search = make_resource_search_hop(
        ctx,
        mode="COLLECT",
        score=_COLLECT_SCORE,
        reason="search_collect_local",
        ai_state=base_state,
    )
    if search is not None:
        return search

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
    """Return a radar decision when the current viewport hasn't been scanned.

    The COLLECT-mode equivalent of HUNT's ``scan_on_landing``: fired
    once per fresh teleport landing in COLLECT mode, before any
    pickup logic runs. The gate is "the current viewport has zero
    tiles in ``scanned_tiles``" -- once a radar fires (extras mark
    the full viewport, free marks 25 around the tank), at least one
    tile carries a live mark, this returns ``None`` on subsequent
    ticks, and the normal pickup -> forage -> hop cascade takes
    over. Pairs with the 0x5A container lift: 0x5A enumerates only
    the tiles the server's viewport patch touches; the landing radar
    fills in the rest so the planner picks an optimal pickup order
    on the next tick rather than committing to the first 0x5A entry.

    Args:
        ctx: Decision context.
        base_state: Base AI state to rewrite for the produced command.

    Returns:
        ``forage_radar``-shaped decision, or ``None`` when the
        viewport already has any scan coverage.
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    scanned_tiles = ctx.world["scanned_tiles"]
    now_ms = ctx.timestamp_ms
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if is_tile_covered(scanned_tiles, x, y, now_ms):
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
        clear_resource_target(base_state),
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


def _select_and_pickup_fuel(
    ctx: DecideCtx,
    base_state: AIStateDict,
) -> TickDecisionDict | None:
    if is_fuel_at_learned_capacity(ctx.fuel):
        return None
    selection = select_fuel_target(ctx)
    if selection is None:
        return None
    container, command = selection
    target_x = container["x"]
    target_y = container["y"]
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
        now_ms=ctx.timestamp_ms,
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
        now_ms=ctx.timestamp_ms,
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
