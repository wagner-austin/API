"""Walk-or-teleport move construction.

The one owner of "how do I get to that tile": the walk/teleport
choice, terrain-aware and terrain-blind variants, approach tiles, and
the pickup and fallback commands. Exploration target choice is
:mod:`tankpit_bot.bot.ai.movement_exploration`.
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
    find_attainable_landing_tile,
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
from tankpit_bot.state.occupancy import is_tank_body_present


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

    if ctx.ws.is_move_target_failed(tx, ty, ctx.timestamp_ms):
        emit_ai("skipping failed move target (%d,%d)", tx, ty)
        return None

    if ctx.ws.recent_own_mine_hit(ctx.timestamp_ms):
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

    This fallback serves a PICKUP approach, so the landing must be
    attainable, not merely legal: a known mine on the tile displaces
    the landing outside the transfer's on-or-cardinal reach every time
    ([[mine-mechanics]]; session bot-20260805-173034 re-aimed 534
    displaced teleports at one mined tile). A mine-denied target
    returns ``None`` — the lock holds, and the clearance step or the
    ``unservable`` release resolves it. The transport-flavored
    teleports (``_mine_flip_teleport``, combat aims) deliberately keep
    plain legality: they only need to arrive NEAR, and displacement is
    acceptable there.
    """
    landing = find_attainable_landing_tile(terrain, tx, ty)
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
    "walk_or_teleport",
]
