"""Movement and exploration planning for AI strategy.

Handles walk, teleport, and exploration commands. All functions operate on a
``DecideCtx`` and return ``BotCommand | None``.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport,
    local_actionable_bounds,
    teleport_fuel_cost_to,
)
from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.ai.equipment_search import find_teleport_landing_tile
from tankpit_bot.bot.ai.ferry import (
    SurfaceRouteTerrain,
    clamp_move_target_at_surface_transition,
    is_riding_ferry,
)
from tankpit_bot.bot.ai.reachability import (
    is_collection_reachable_in_viewport,
    is_move_reachable_in_viewport,
)
from tankpit_bot.bot.types import (
    BotCommand,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_teleport_command,
)
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.sniffer.world_state import (
    is_move_target_failed,
    is_scan_viewport_failed,
    recent_own_mine_hit,
)
from tankpit_bot.state.occupancy import is_tank_body_present
from tankpit_bot.state.scan_coverage import is_viewport_fully_covered
from tankpit_bot.state.types import viewport_scan_key


def walk_or_teleport(
    ctx: DecideCtx,
    tx: int,
    ty: int,
    *,
    pickup_kind: str | None = "equipment",
) -> BotCommand | None:
    """Plan a direct walk or teleport for a target.

    The game can walk directly across clear ground, but does not reliably path
    around terrain obstacles. This helper therefore prefers:
    1. direct move/pickup when the current viewport can execute the path
    2. teleport fallback when no viewport-contained walk route exists

    Rejects move destinations that are occupied by enemy tanks or that
    recently failed (stalled and timed out).

    Args:
        ctx: Decision context.
        tx: Target X coordinate.
        ty: Target Y coordinate.
        pickup_kind: Resource kind for pickup commands, or None for pure moves.

    Returns:
        Planned command, or None if no executable route exists.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]

    if is_move_target_failed(tx, ty, ctx.timestamp_ms):
        emit_ai("skipping failed move target (%d,%d)", tx, ty)
        return None

    if recent_own_mine_hit(ctx.timestamp_ms):
        # User movement doctrine (2026-07-30): "walk to targets or
        # containers in viewport but if we hit a mine teleport to
        # target or container. then resume walking within viewport."
        # A walk-over just cost 45 and arrested the move, which means
        # unrevealed mines sit on the walking route; the teleport
        # landing is mine-immune by the displacement law, and the
        # flip window expires so walking resumes afterwards.
        flip = _mine_flip_teleport(ctx, tx, ty)
        if flip is not None:
            emit_ai(
                "mine walk-over flip: teleporting to (%d,%d) instead of re-walking",
                tx,
                ty,
            )
            return flip

    if ctx.terrain is not None:
        return _walk_or_teleport_with_terrain(ctx, tx, ty, sx, sy, pickup_kind=pickup_kind)
    return _walk_or_teleport_without_terrain(ctx, tx, ty, pickup_kind=pickup_kind)


def _mine_flip_teleport(ctx: DecideCtx, tx: int, ty: int) -> BotCommand | None:
    """Build the post-mine-hit teleport approach to a destination.

    Args:
        ctx: Decision context.
        tx: Destination X.
        ty: Destination Y.

    Returns:
        Teleport command to the destination's landing tile, or
        ``None`` when no landing exists or the hop is unaffordable --
        the caller then falls back to walking (one more 45-fuel risk
        beats stranding the tank).
    """
    if ctx.terrain is None:
        return None
    landing = find_teleport_landing_tile(ctx.terrain, tx, ty)
    if landing is None:
        return None
    landing_x, landing_y = landing
    if not can_afford_teleport(ctx, landing_x, landing_y, reserve_fuel=0):
        return None
    return make_teleport_command(landing_x, landing_y)


def is_pickup_target_actionable(ctx: DecideCtx, tx: int, ty: int) -> bool:
    """Return True when a target is inside the visible viewport.

    Args:
        ctx: Decision context with current viewport.
        tx: Target X coordinate.
        ty: Target Y coordinate.

    Returns:
        True if the target is inside the visible viewport.
    """
    left, top, right, bottom = local_actionable_bounds(ctx)
    return left <= tx <= right and top <= ty <= bottom


def select_exploration_command(
    ctx: DecideCtx,
    *,
    candidate_offset: int = 0,
) -> tuple[int, int, BotCommand] | None:
    """Return the first executable exploration step inside the current viewport.

    Exploration is used when the bot wants fresh information but map/radar
    cannot be used immediately. The search stays inside the visible viewport
    and prefers tiles on the edges that are most likely to reveal a
    new viewport next.

    Args:
        ctx: Decision context with viewport, terrain, and fuel state.
        candidate_offset: Rotation offset into the ordered candidate list.

    Returns:
        Tuple of ``(x, y, command)`` for the first executable exploration
        target, or ``None`` when no exploration command can be executed.
    """
    for candidate_x, candidate_y in viewport_exploration_candidates(
        ctx,
        candidate_offset=candidate_offset,
    ):
        command = walk_or_teleport(ctx, candidate_x, candidate_y, pickup_kind=None)
        if command is None:
            continue
        return (candidate_x, candidate_y, command)
    emit_ai("no executable exploration target in current viewport")
    return None


def viewport_exploration_candidates(
    ctx: DecideCtx,
    *,
    candidate_offset: int = 0,
) -> list[tuple[int, int]]:
    """Return ordered exploration targets on the visible viewport boundary.

    Uses the actual viewport bounds rather than assuming the player is
    centered. The player moves freely inside the fixed viewport frame; it only
    recenters when the player reaches the edge. Exploration should therefore
    prefer the real visible edge while trying multiple edge-aligned candidates
    before giving up.

    Args:
        ctx: Decision context with viewport and self position.
        candidate_offset: Rotation offset into the ordered candidate list.

    Returns:
        Ordered unique candidate coordinates inside the visible viewport.
    """
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    left, top, right, bottom = local_actionable_bounds(ctx)
    preferred_x = right if sx < 128 else left
    preferred_y = bottom if sy < 128 else top
    alternate_x = left if preferred_x == right else right
    alternate_y = top if preferred_y == bottom else bottom
    clamped_x = min(max(sx, left), right)
    clamped_y = min(max(sy, top), bottom)
    middle_x = (left + right) // 2
    middle_y = (top + bottom) // 2

    ordered = [
        (preferred_x, preferred_y),
        (preferred_x, clamped_y),
        (clamped_x, preferred_y),
        (preferred_x, alternate_y),
        (alternate_x, preferred_y),
        (preferred_x, middle_y),
        (middle_x, preferred_y),
        (alternate_x, clamped_y),
        (clamped_x, alternate_y),
        (alternate_x, alternate_y),
        (alternate_x, middle_y),
        (middle_x, alternate_y),
    ]
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int]] = []
    for candidate_x, candidate_y in ordered:
        candidate = (candidate_x, candidate_y)
        if candidate == (sx, sy):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    if not candidates:
        return candidates
    ranked_candidates = [
        (_candidate_priority_for_ctx(ctx, candidate), candidate) for candidate in candidates
    ]
    ranked_candidates.sort(key=_ranked_exploration_key)
    candidates = [candidate for _, candidate in ranked_candidates]
    offset = candidate_offset % len(candidates)
    return candidates[offset:] + candidates[:offset]


# =============================================================================
# Internal helpers
# =============================================================================


def _approach_target(ctx: DecideCtx, tx: int, ty: int) -> tuple[int, int]:
    """Clamp an off-viewport target to the visible viewport edge.

    Args:
        ctx: Decision context with current viewport.
        tx: Target X coordinate.
        ty: Target Y coordinate.

    Returns:
        Edge approach tile inside the visible viewport.
    """
    left, top, right, bottom = local_actionable_bounds(ctx)
    approach_x = min(max(tx, left), right)
    approach_y = min(max(ty, top), bottom)
    return (approach_x, approach_y)


def _select_walkable_approach_tile(
    ctx: DecideCtx,
    tx: int,
    ty: int,
) -> tuple[int, int] | None:
    """Return a passable, walk-reachable viewport tile facing an off-viewport target.

    The geometric clamp alone is terrain-blind: live run 20260610-000x
    rejected every known equipment container because the single projected
    edge tile happened to be rock or water (for example targets west of
    the viewport all clamped onto the x=185 rock ridge). This selector
    starts from the clamp and scans outward along the facing viewport
    edge for the nearest tile that is passable AND walk-reachable,
    keeping known off-viewport containers approachable on broken ground.

    Args:
        ctx: Decision context with viewport and terrain.
        tx: Real target X coordinate (may be off-viewport).
        ty: Real target Y coordinate (may be off-viewport).

    Returns:
        Best approach tile, or ``None`` when no facing-edge tile is
        currently walkable.
    """
    terrain = ctx.terrain
    clamp_x, clamp_y = _approach_target(ctx, tx, ty)
    if terrain is None:
        return (clamp_x, clamp_y)
    left, top, right, bottom = local_actionable_bounds(ctx)
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    # Callers only reach this for off-viewport targets, so at least one
    # axis is clamped: scan along the facing edge of whichever axis it is.
    if tx < left or tx > right:
        candidates = [(clamp_x, y) for y in range(top, bottom + 1)]
    else:
        candidates = [(x, clamp_y) for x in range(left, right + 1)]

    def _distance_from_clamp(tile: tuple[int, int]) -> int:
        """Order candidates by closeness to the geometric projection."""
        return abs(tile[0] - clamp_x) + abs(tile[1] - clamp_y)

    candidates.sort(key=_distance_from_clamp)
    for cx, cy in candidates:
        if (cx, cy) == (sx, sy):
            continue
        if not terrain.is_passable(cx, cy):
            continue
        if not is_move_reachable_in_viewport(
            ctx.world,
            terrain,
            sx,
            sy,
            cx,
            cy,
        ):
            continue
        return (cx, cy)
    return None


def _predicted_viewport_origin_for_target(
    ctx: DecideCtx,
    target_x: int,
    target_y: int,
) -> tuple[int, int]:
    """Return the viewport origin likely after reaching a target tile."""
    viewport = ctx.world["viewport"]
    max_left = max(0, 256 - viewport["width"])
    max_top = max(0, 256 - viewport["height"])
    return (
        min(max(target_x - (viewport["width"] // 2), 0), max_left),
        min(max(target_y - (viewport["height"] // 2), 0), max_top),
    )


def _exploration_priority(
    ctx: DecideCtx,
    target_x: int,
    target_y: int,
) -> tuple[bool, bool, bool, int]:
    """Rank exploration targets by how much fresh scan space they expose."""
    next_left, next_top = _predicted_viewport_origin_for_target(ctx, target_x, target_y)
    current_viewport = ctx.world["viewport"]
    current_key = viewport_scan_key(current_viewport["left"], current_viewport["top"])
    next_key = viewport_scan_key(next_left, next_top)
    is_current = next_key == current_key
    is_failed = is_scan_viewport_failed(next_left, next_top, ctx.timestamp_ms)
    is_scanned = is_viewport_fully_covered(
        ctx.world["scanned_tiles"],
        next_left,
        next_top,
        next_left + current_viewport["width"] - 1,
        next_top + current_viewport["height"] - 1,
        ctx.timestamp_ms,
    )
    reveal_distance = abs(target_x - ctx.self_state["x"]) + abs(target_y - ctx.self_state["y"])
    return (is_current, is_failed, is_scanned, -reveal_distance)


def _candidate_priority_for_ctx(
    ctx: DecideCtx,
    candidate: tuple[int, int],
) -> tuple[bool, bool, bool, int]:
    """Return a typed sort key wrapper for exploration candidates."""
    return _exploration_priority(ctx, candidate[0], candidate[1])


def _ranked_exploration_key(
    item: tuple[tuple[bool, bool, bool, int], tuple[int, int]],
) -> tuple[bool, bool, bool, int]:
    """Return the priority part of a ranked exploration candidate."""
    return item[0]


def _is_occupied_by_enemy(ctx: DecideCtx, x: int, y: int) -> bool:
    """Return True when a tile is occupied by an enemy tank's body.

    Body presence is the occupancy law
    (:func:`~tankpit_bot.state.occupancy.is_tank_body_present`: not
    self, position ever observed, viewport-fresh); the enemy-team
    filter is this call site's own policy on top -- allies do not
    block movement or pickup. Before the lift this re-derived
    presence with no freshness or position gate, so a login-roster
    phantom or a long-departed tank could veto a tile.

    Args:
        ctx: Decision context with current world state.
        x: Tile X coordinate to check.
        y: Tile Y coordinate to check.

    Returns:
        True if the tile is occupied by an enemy body.
    """
    self_team = ctx.self_state["team"]
    return any(
        tank["x"] == x
        and tank["y"] == y
        and tank["team"] != self_team
        and is_tank_body_present(tank, ctx.timestamp_ms)
        for tank in ctx.filtered["tanks"].values()
    )


def _is_occupied_by_mine(ctx: DecideCtx, x: int, y: int) -> bool:
    """Return True when a tile is occupied by a known mine.

    Args:
        ctx: Decision context with current world state.
        x: Tile X coordinate to check.
        y: Tile Y coordinate to check.

    Returns:
        True if the tile is occupied by a known mine.
    """
    return f"{x},{y}" in hostile_mines(ctx.world)


def _walk_or_teleport_with_terrain(
    ctx: DecideCtx,
    tx: int,
    ty: int,
    sx: int,
    sy: int,
    *,
    pickup_kind: str | None,
) -> BotCommand | None:
    """Resolve movement when terrain/pathfinding is available.

    Pickups are dispatched as ONE ``pickup_*`` command (the JS client
    does the same: clicking a container is a single long-press; the
    server walks the tank and completes the pickup). The dispatch is
    gated on a walk path existing inside the current viewport -- a
    pickup with no walk path is unreachable from where the bot
    stands, and the user contract is to skip such containers and
    let the search-hop relocate (no teleport-to-container fallback).
    Plain moves keep their teleport fallback for combat-approach and
    exploration.
    """
    terrain = ctx.terrain
    assert terrain is not None  # caller guarantees this
    if pickup_kind is not None:
        if not is_pickup_target_actionable(ctx, tx, ty):
            return None
        # A pickup is ONE server-routed click, and one command never
        # chains surfaces (user contract 2026-07-19/20): the route
        # must exist on the tank's CURRENT surface -- plain ground on
        # land, water while riding a ferry (a floating container
        # picks up normally from the ferry). Cross-surface targets
        # need the two-action embark/disembark dance first.
        riding = is_riding_ferry(ctx.world)
        if not is_collection_reachable_in_viewport(
            ctx.world,
            SurfaceRouteTerrain(terrain, water=riding),
            sx,
            sy,
            tx,
            ty,
        ):
            if riding:
                # Two-action contract: disembark first (a piloted move
                # bounded at the first land tile), then next tick's
                # replan dispatches the pickup from solid ground.
                emit_ai(
                    "pickup at (%d,%d) needs land routing -- disembarking first",
                    tx,
                    ty,
                )
                return _surface_clamped_move(ctx, terrain, sx, sy, tx, ty)
            return None
        emit_ai("dispatching %s pickup at (%d,%d)", pickup_kind, tx, ty)
        return _make_pickup_command(pickup_kind, tx, ty)
    if not is_pickup_target_actionable(ctx, tx, ty):
        return _approach_command(ctx, tx, ty)
    if is_move_reachable_in_viewport(
        ctx.world,
        terrain,
        sx,
        sy,
        tx,
        ty,
    ):
        return _surface_clamped_move(ctx, terrain, sx, sy, tx, ty)
    return _teleport_fallback_command(ctx, terrain, tx, ty)


def _surface_clamped_move(
    ctx: DecideCtx,
    terrain: TerrainMapProtocol,
    sx: int,
    sy: int,
    tx: int,
    ty: int,
) -> BotCommand | None:
    """Issue a direct move bounded at the first surface transition.

    Boarding a ferry and stepping from ferry/water onto land each
    consume one action-queue slot: the server stops the tank on the
    transition tile, so a command planned past it would stall against
    its own target. When the path crosses a surface boundary the
    command becomes a plain move to that boundary tile and the next
    tick replans the remainder.

    Pickup paths bypass this helper entirely -- they dispatch a
    single ``pickup_fuel`` / ``pickup_equipment`` command and let
    the server route the tank; only plain moves need surface
    clamping.

    Args:
        ctx: Decision context.
        terrain: Ferry-aware terrain view used for planning.
        sx: Starting X coordinate.
        sy: Starting Y coordinate.
        tx: Requested target X coordinate.
        ty: Requested target Y coordinate.

    Returns:
        Planned command, or None when the clamped tile is occupied.
    """
    clamp_x, clamp_y = clamp_move_target_at_surface_transition(
        ctx.world,
        terrain,
        sx,
        sy,
        tx,
        ty,
    )
    if (clamp_x, clamp_y) != (tx, ty):
        emit_ai(
            "surface transition at (%d,%d) bounds move toward (%d,%d)",
            clamp_x,
            clamp_y,
            tx,
            ty,
        )
        return _direct_move_command(ctx, clamp_x, clamp_y)
    return _direct_move_command(ctx, tx, ty)


def _walk_or_teleport_without_terrain(
    ctx: DecideCtx,
    tx: int,
    ty: int,
    *,
    pickup_kind: str | None,
) -> BotCommand | None:
    """Resolve movement when only local occupancy checks are available.

    Pickups still require the target to be inside the visible
    viewport (the JS click semantic), but without a terrain map
    there is no walk-path check. Plain moves keep their off-viewport
    approach fallback.
    """
    if pickup_kind is not None:
        if not is_pickup_target_actionable(ctx, tx, ty):
            return None
        return _make_pickup_command(pickup_kind, tx, ty)
    if not is_pickup_target_actionable(ctx, tx, ty):
        return _approach_command(ctx, tx, ty)
    if _is_occupied_by_enemy(ctx, tx, ty):
        emit_ai("move target (%d,%d) is occupied by enemy", tx, ty)
        return None
    if _is_occupied_by_mine(ctx, tx, ty):
        emit_ai("move target (%d,%d) is occupied by mine", tx, ty)
        return None
    return make_move_command(tx, ty)


def _approach_command(
    ctx: DecideCtx,
    tx: int,
    ty: int,
) -> BotCommand | None:
    """Return an approach command for an off-viewport plain-move target.

    Prefers a cheap walk to a terrain-validated viewport-edge tile facing
    the target. When no facing-edge tile is walkable, teleports to the
    landing tile near the REAL target instead -- the bot knows the exact
    coordinates, and abandoning a known target because one edge tile is
    rock wastes radar and fuel on blind re-searches.

    Pickups never reach this helper: the user contract (2026-06-26)
    forbids teleport-to-container, so off-viewport pickups are skipped
    upstream.
    """
    approach = _select_walkable_approach_tile(ctx, tx, ty)
    if approach is not None:
        approach_x, approach_y = approach
        emit_ai(
            "move target (%d,%d) is outside viewport, approaching via (%d,%d)",
            tx,
            ty,
            approach_x,
            approach_y,
        )
        walked = walk_or_teleport(ctx, approach_x, approach_y, pickup_kind=None)
        if walked is not None:
            return walked
    if ctx.terrain is None:
        return None
    emit_ai(
        "no walkable approach edge for move target (%d,%d), teleporting to it directly",
        tx,
        ty,
    )
    return _teleport_fallback_command(ctx, ctx.terrain, tx, ty)


def _direct_move_command(
    ctx: DecideCtx,
    tx: int,
    ty: int,
) -> BotCommand | None:
    """Return a direct move command when the straight path is clear.

    Pickup-kind targets dispatch their own ``pickup_*`` command
    upstream; this helper only emits plain moves.
    """
    if not is_pickup_target_actionable(ctx, tx, ty):
        emit_ai("direct target (%d,%d) is outside viewport", tx, ty)
        return None
    if _is_occupied_by_enemy(ctx, tx, ty):
        emit_ai("move target (%d,%d) is occupied by enemy", tx, ty)
        return None
    if _is_occupied_by_mine(ctx, tx, ty):
        emit_ai("move target (%d,%d) is occupied by mine", tx, ty)
        return None
    return make_move_command(tx, ty)


def _make_pickup_command(kind: str, tx: int, ty: int) -> BotCommand:
    """Return the protocol-correct pickup command for a resource kind.

    Args:
        kind: Resource kind ("fuel" or "equipment").
        tx: Target X coordinate.
        ty: Target Y coordinate.

    Returns:
        The pickup command for the specified kind.

    Raises:
        ValueError: If kind is not "fuel" or "equipment".
    """
    if kind == "fuel":
        return make_pickup_fuel_command(tx, ty)
    if kind == "equipment":
        return make_pickup_equipment_command(tx, ty)
    raise ValueError(f"Unknown pickup kind: {kind}")


def _teleport_fallback_command(
    ctx: DecideCtx,
    terrain: TerrainMapProtocol,
    tx: int,
    ty: int,
) -> BotCommand | None:
    """Return a teleport command for a terrain-blocked target when possible.

    The server handles displacement when the landing tile is occupied or
    impassable, so ``find_teleport_landing_tile`` always returns the goal
    for in-bounds targets. The only rejection is when the teleport is
    unaffordable.
    """
    landing = find_teleport_landing_tile(terrain, tx, ty)
    if landing is None:
        return None
    lx, ly = landing
    if not can_afford_teleport(ctx, lx, ly):
        emit_ai(
            "cannot afford teleport fallback to (%d,%d) for (%d,%d) (fuel=%d cost=%d)",
            lx,
            ly,
            tx,
            ty,
            ctx.fuel,
            teleport_fuel_cost_to(ctx, lx, ly),
        )
        return None
    emit_ai("terrain blocked, teleporting to target at (%d,%d)", lx, ly)
    return make_teleport_command(lx, ly)


__all__ = [
    "is_pickup_target_actionable",
    "select_exploration_command",
    "viewport_exploration_candidates",
    "walk_or_teleport",
]
