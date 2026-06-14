"""Shared resource-search helpers for recovery-oriented AI modes.

This module centralizes the local sector-hop logic used by both fuel and
equipment recovery so durable owners do not duplicate teleport-search
behavior.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport_search,
    clear_resource_target,
    make_decision,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.equipment import (
    SCAN_COVERAGE_TTL_MS as _SCAN_COVERAGE_TTL_MS,
)
from tankpit_bot.bot.ai.equipment import (
    is_area_scanned,
)
from tankpit_bot.bot.ai.movement import select_exploration_command
from tankpit_bot.bot.ai.types import AIStateDict, BehaviorMode
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_map_open_command, make_teleport_command
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.state.types import coord_key, parse_coord_key

_CARDINAL_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)

# Hop rings beyond this wrap back to ring 1. Without the cap the patrol
# index grows for the whole session and hop distances scale with it:
# live run 20260610-000x reached 90-tile hops costing 540 fuel, which
# the owner could no longer afford -- it then had no legal action left.
_MAX_SEARCH_RINGS = 3

# A hop whose clamped target moves the tank fewer tiles than this is a
# degenerate re-visit (the map-edge clamp collapsed it onto the current
# position) and never worth a teleport.
_MIN_HOP_DISPLACEMENT = 4


def _hop_target_for_cycle(ctx: DecideCtx, cycle: int) -> tuple[int, int]:
    """Return the clamped hop target for a position in the ring cycle.

    Args:
        ctx: Decision context.
        cycle: Position within the ``directions x rings`` cycle.

    Returns:
        Clamped ``(target_x, target_y)`` for the cycle position.
    """
    index = cycle % len(_CARDINAL_DIRECTIONS)
    dx, dy = _CARDINAL_DIRECTIONS[index]
    ring = 1 + (cycle // len(_CARDINAL_DIRECTIONS))
    distance = ctx.config["equip_search_hop_distance"] * ring
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    target_x = max(1, min(254, sx + dx * distance))
    target_y = max(1, min(254, sy + dy * distance))
    return (target_x, target_y)


def _is_worthwhile_hop(ctx: DecideCtx, target_x: int, target_y: int) -> bool:
    """Return True when a hop target reveals ground worth scanning.

    Two degenerate cases are rejected. A map-edge clamp can collapse
    the target onto the current position (live run 20260610: corner at
    (1,254) cycled hops to (1,254)/(3,254)/(5,254) forever). And a
    target whose viewport is already covered by a fresh scan re-radars
    ground the bot just saw.

    Args:
        ctx: Decision context.
        target_x: Clamped hop target X coordinate.
        target_y: Clamped hop target Y coordinate.

    Returns:
        True when the hop moves far enough and lands on uncovered ground.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    if abs(target_x - sx) + abs(target_y - sy) < _MIN_HOP_DISPLACEMENT:
        return False
    viewport = ctx.world["viewport"]
    landing_left = target_x - viewport["width"] // 2
    landing_top = target_y - viewport["height"] // 2
    return not is_area_scanned(ctx.world, landing_left, landing_top, ctx.timestamp_ms)


def local_resource_search_hop(ctx: DecideCtx) -> tuple[int, int, int]:
    """Compute the next local sector hop for resource recovery.

    Starting from the persisted patrol index, the full ring cycle is
    searched for the first hop that escapes the current position and
    lands on unscanned ground. When every cycle position is degenerate
    or covered, the raw indexed hop is returned so the owner always has
    a target -- coverage expires, so this self-heals within the scan
    TTL.

    Args:
        ctx: Decision context.

    Returns:
        Tuple of ``(target_x, target_y, next_patrol_index)``.
    """
    raw_index = ctx.ai_state["patrol_waypoint_index"]
    cycle_length = len(_CARDINAL_DIRECTIONS) * _MAX_SEARCH_RINGS
    for offset in range(cycle_length):
        cycle = (raw_index + offset) % cycle_length
        target_x, target_y = _hop_target_for_cycle(ctx, cycle)
        if _is_worthwhile_hop(ctx, target_x, target_y):
            return (target_x, target_y, raw_index + offset + 1)
    target_x, target_y = _hop_target_for_cycle(ctx, raw_index % cycle_length)
    return (target_x, target_y, raw_index + 1)


def is_recently_attempted(
    attempted: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
    *,
    ttl_ms: int,
) -> bool:
    """Return True when a coordinate carries a live attempt mark.

    Args:
        attempted: Attempt marks keyed by "x,y" with dispatch timestamps.
        x: Target X coordinate.
        y: Target Y coordinate.
        now_ms: Current timestamp for TTL evaluation.
        ttl_ms: Mark lifetime in milliseconds.

    Returns:
        True if the coordinate was attempted within the TTL.
    """
    attempted_ms = attempted.get(coord_key(x, y))
    return attempted_ms is not None and now_ms - attempted_ms <= ttl_ms


def record_attempt_mark(
    attempted: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
    *,
    ttl_ms: int,
) -> dict[str, int]:
    """Return attempt marks with expired entries pruned and (x, y) recorded.

    Args:
        attempted: Attempt marks keyed by "x,y" with dispatch timestamps.
        x: Target X coordinate to record.
        y: Target Y coordinate to record.
        now_ms: Dispatch timestamp recorded for the new mark.
        ttl_ms: Mark lifetime in milliseconds used for pruning.

    Returns:
        New attempt-mark mapping.
    """
    pruned = {
        key: marked_ms for key, marked_ms in attempted.items() if now_ms - marked_ms <= ttl_ms
    }
    pruned[coord_key(x, y)] = now_ms
    return pruned


def select_fuel_dot_hop(ctx: DecideCtx) -> tuple[int, int] | None:
    """Pick the nearest worthwhile, affordable fuel-dot relocation target.

    The map's fuel-dot atlas (see ``WorldStateDict.map_fuel_dots``) marks
    where fuel containers were as of the server's MAP_DATA snapshot.
    Candidates are scanned nearest-first; a dot inside freshly scanned
    ground or within the degenerate-hop displacement is skipped because
    local truth already covers it. Teleport cost is monotone in
    distance, so the first worthwhile dot that is unaffordable ends the
    scan -- every farther dot costs more.

    Args:
        ctx: Decision context.

    Returns:
        ``(x, y)`` of the chosen fuel dot, or ``None`` when the atlas is
        empty, fully covered by fresh scans, or unaffordable.
    """
    dots = ctx.world["map_fuel_dots"]
    if not dots:
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    candidates = sorted(
        (abs(x - sx) + abs(y - sy), y, x) for x, y in (parse_coord_key(key) for key in dots)
    )
    for _distance, y, x in candidates:
        if not _is_worthwhile_hop(ctx, x, y):
            continue
        if not can_afford_teleport_search(ctx, x, y):
            return None
        return (x, y)
    return None


def select_fuel_dot_walk_targets(ctx: DecideCtx) -> list[tuple[int, int]]:
    """Return fuel-dot atlas targets sorted by distance, nearest first.

    Like :func:`select_fuel_dot_hop` but returns ALL worthwhile dots
    (not just the first affordable one) and does NOT filter by teleport
    affordability -- the caller decides whether to walk or teleport.
    Dots that have already been attempted within the scan coverage TTL
    are excluded so the walker does not revisit recently failed tiles.

    Args:
        ctx: Decision context.

    Returns:
        List of ``(x, y)`` tuples sorted nearest-first, possibly empty.
    """
    dots = ctx.world["map_fuel_dots"]
    if not dots:
        return []
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    candidates = sorted(
        (abs(x - sx) + abs(y - sy), y, x) for x, y in (parse_coord_key(key) for key in dots)
    )
    result: list[tuple[int, int]] = []
    for _distance, y, x in candidates:
        if abs(x - sx) + abs(y - sy) < _MIN_HOP_DISPLACEMENT:
            continue
        if is_recently_attempted(
            ctx.ai_state["attempted_fuel_dots"],
            x,
            y,
            ctx.timestamp_ms,
            ttl_ms=_SCAN_COVERAGE_TTL_MS,
        ):
            continue
        result.append((x, y))
    return result


def emit_fuel_dot_hop_diagnostic(ctx: DecideCtx, dot_x: int, dot_y: int) -> None:
    """Emit a diagnostic event for a fuel-dot hop or refuel teleport.

    Args:
        ctx: Decision context.
        dot_x: Target fuel-dot X coordinate.
        dot_y: Target fuel-dot Y coordinate.
    """
    emit_diagnostic(
        diagnostic_kind="fuel_dot_hop",
        target_x=dot_x,
        target_y=dot_y,
        self_x=ctx.self_state["x"],
        self_y=ctx.self_state["y"],
        dots_known=len(ctx.world["map_fuel_dots"]),
        fuel=ctx.fuel,
    )


def make_resource_search_hop(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    reason: str,
    failure_count: int | None = None,
    ai_state: AIStateDict | None = None,
    fuel_dot_guided: bool = False,
) -> TickDecisionDict | None:
    """Create a teleport-search decision for recovery behavior.

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the hop.
        reason: Behavior reason label.
        failure_count: Optional consecutive-failure count for recovery search.
        ai_state: Optional AI state base to rewrite before returning.
        fuel_dot_guided: When True, prefer relocating to the nearest
            worthwhile fuel dot from the map atlas over the blind ring
            patrol. Fuel recovery sets this; equipment recovery must not
            (dots never mark equipment -- verified 0/184 on 2026-06-11).

    Returns:
        Recovery teleport decision, or ``None`` when fuel is too low to hop.
    """
    dot_target = select_fuel_dot_hop(ctx) if fuel_dot_guided else None
    if dot_target is not None:
        target_x, target_y = dot_target
        # A dot hop does not consume a ring-patrol position.
        next_index = ctx.ai_state["patrol_waypoint_index"]
        emit_diagnostic(
            diagnostic_kind="fuel_dot_hop",
            target_x=target_x,
            target_y=target_y,
            self_x=ctx.self_state["x"],
            self_y=ctx.self_state["y"],
            dots_known=len(ctx.world["map_fuel_dots"]),
            fuel=ctx.fuel,
        )
    else:
        target_x, target_y, next_index = local_resource_search_hop(ctx)
    if not can_afford_teleport_search(ctx, target_x, target_y):
        emit_ai(
            "cannot afford %s hop to (%d,%d) (fuel=%d cost=%d reserve=%d)",
            reason,
            target_x,
            target_y,
            ctx.fuel,
            teleport_fuel_cost_to(ctx, target_x, target_y),
            ctx.config["hunt_min_fuel"],
        )
        return None

    base_state = ctx.base if ai_state is None else ai_state
    cleared = clear_resource_target(base_state)

    hop_kind = "fuel-dot" if dot_target is not None else "local resource"
    if failure_count is None:
        emit_ai(
            "%s hop to (%d,%d) (dual=%d homing=%d radar=%d)",
            hop_kind,
            target_x,
            target_y,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
        )
        updated_state = AIStateDict(
            **{
                **cleared,
                "patrol_waypoint_index": next_index,
            }
        )
    else:
        next_failures = failure_count + 1
        emit_ai(
            "%s hop to (%d,%d) (dual=%d homing=%d radar=%d attempt=%d)",
            hop_kind,
            target_x,
            target_y,
            ctx.inventory["dual_shots"]["count"],
            ctx.inventory["homing_shots"]["count"],
            ctx.inventory["extra_radars"]["count"],
            next_failures,
        )
        updated_state = AIStateDict(
            **{
                **cleared,
                "patrol_waypoint_index": next_index,
                "equipment_search_failures": next_failures,
            }
        )

    return make_decision(
        make_teleport_command(target_x, target_y),
        mode,
        score,
        target_x,
        target_y,
        reason,
        updated_state,
        ctx.equip,
    )


def make_recovery_edge_decision(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    reason: str,
    ai_state: AIStateDict,
) -> TickDecisionDict | None:
    """Walk to a viewport edge when teleport-search is not affordable.

    Edge walking is the cheap recovery fallback: it costs almost no fuel,
    reveals a fresh viewport, and keeps the recovery owner acting instead
    of stalling when the hop planner cannot afford a teleport.

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the edge walk.
        reason: Behavior reason label.
        ai_state: Base AI state to rewrite.

    Returns:
        Edge-walk decision, or ``None`` when no viewport edge tile is
        currently walkable.
    """
    exploration = select_exploration_command(
        ctx,
        candidate_offset=ai_state["patrol_waypoint_index"],
    )
    if exploration is None:
        return None
    edge_x, edge_y, edge_command = exploration
    emit_ai("recovery edge walk to (%d,%d) (fuel=%d)", edge_x, edge_y, ctx.fuel)
    return make_decision(
        edge_command,
        mode,
        score,
        edge_x,
        edge_y,
        reason,
        AIStateDict(
            **{
                **ai_state,
                "patrol_waypoint_index": ai_state["patrol_waypoint_index"] + 1,
            }
        ),
        ctx.equip,
    )


def make_recovery_map_intel_decision(
    ctx: DecideCtx,
    *,
    mode: BehaviorMode,
    score: int,
    reason: str,
    ai_state: AIStateDict,
) -> TickDecisionDict:
    """Open the map as the terminal recovery action when fully boxed in.

    With the terrain-aware approach and the capped search ring in place
    this state should be near-unreachable, so reaching it is ALWAYS
    surfaced as a loud ``recovery_boxed_in`` DIAGNOSTIC that the issue
    report promotes to a top-level issue. The action itself is the only
    information-gathering move the game still offers here: the map costs
    nothing, refreshes tank intelligence, and the HFSM map_open gate
    throttles replanning until the MAP_DATA response is processed.
    Raising instead would kill the bot process mid-game (live run
    20260610-000x).

    Args:
        ctx: Decision context.
        mode: Behavior mode label for the decision.
        score: Behavior score for the map-intel action.
        reason: Behavior reason label.
        ai_state: Base AI state to carry through unchanged.

    Returns:
        Map-open decision; this function always produces an action.
    """
    emit_ai("recovery boxed in - opening map for fresh intel (fuel=%d)", ctx.fuel)
    emit_diagnostic(
        diagnostic_kind="recovery_boxed_in",
        behavior_mode=mode,
        fuel=ctx.fuel,
        self_x=ctx.self_state["x"],
        self_y=ctx.self_state["y"],
        patrol_waypoint_index=ctx.ai_state["patrol_waypoint_index"],
    )
    return make_decision(
        make_map_open_command(),
        mode,
        score,
        0,
        0,
        reason,
        ai_state,
        ctx.equip,
    )


__all__ = [
    "emit_fuel_dot_hop_diagnostic",
    "is_recently_attempted",
    "local_resource_search_hop",
    "make_recovery_edge_decision",
    "make_recovery_map_intel_decision",
    "make_resource_search_hop",
    "record_attempt_mark",
    "select_fuel_dot_hop",
    "select_fuel_dot_walk_targets",
]
