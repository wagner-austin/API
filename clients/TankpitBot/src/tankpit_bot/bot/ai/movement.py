"""Movement and exploration planning for AI strategy.

Handles walk, waypoint, teleport, and exploration commands. All functions
operate on a ``DecideCtx`` and return ``BotCommand | None``.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport_search,
    local_actionable_bounds,
)
from tankpit_bot.bot.ai.equipment import find_teleport_landing_tile
from tankpit_bot.bot.ai.pathfinding import find_path_segment_target, is_direct_path_clear
from tankpit_bot.bot.types import (
    BotCommand,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_teleport_command,
)
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.sniffer.world_state import is_move_target_failed
from tankpit_bot.state.types import MineStateDict


def walk_or_teleport(
    ctx: DecideCtx,
    tx: int,
    ty: int,
    *,
    pickup_kind: str | None = "equipment",
) -> BotCommand | None:
    """Plan a direct walk, waypointed walk, or teleport for a target.

    The game can walk directly across clear ground, but does not reliably path
    around terrain obstacles. This helper therefore prefers:
    1. direct move/pickup when the straight route is clear
    2. a terrain-aware waypoint along the first A* segment
    3. teleport fallback when no walk route exists

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

    if pickup_kind is None and is_move_target_failed(tx, ty, ctx.timestamp_ms):
        emit_ai("skipping failed move target (%d,%d)", tx, ty)
        return None

    if ctx.terrain is not None:
        return _walk_or_teleport_with_terrain(ctx, tx, ty, sx, sy, pickup_kind=pickup_kind)
    return _walk_or_teleport_without_terrain(ctx, tx, ty, pickup_kind=pickup_kind)


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


def select_exploration_command(ctx: DecideCtx) -> tuple[int, int, BotCommand] | None:
    """Return the first executable exploration step inside the current viewport.

    Exploration is used when the bot wants fresh information but map/radar
    cannot be used immediately. The search stays inside the visible viewport
    and prefers tiles on the edges that are most likely to reveal a
    new viewport next.

    Args:
        ctx: Decision context with viewport, terrain, and fuel state.

    Returns:
        Tuple of ``(x, y, command)`` for the first executable exploration
        target, or ``None`` when no exploration command can be executed.
    """
    for candidate_x, candidate_y in viewport_exploration_candidates(ctx):
        command = walk_or_teleport(ctx, candidate_x, candidate_y, pickup_kind=None)
        if command is None:
            continue
        if command["cmd_type"] == "teleport" and not can_afford_teleport_search(ctx):
            emit_ai(
                "skipping exploration teleport to (%d,%d) - fuel too low (%d)",
                candidate_x,
                candidate_y,
                ctx.fuel,
            )
            continue
        return (candidate_x, candidate_y, command)
    emit_ai("no executable exploration target in current viewport")
    return None


def viewport_exploration_candidates(ctx: DecideCtx) -> list[tuple[int, int]]:
    """Return ordered exploration targets on the visible viewport boundary.

    Uses the actual viewport bounds rather than assuming the player is
    centered. The player moves freely inside the fixed viewport frame; it only
    recenters when the player reaches the edge. Exploration should therefore
    prefer the real visible edge while trying multiple edge-aligned candidates
    before giving up.

    Args:
        ctx: Decision context with viewport and self position.

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
    return candidates


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


def _is_occupied_by_enemy(ctx: DecideCtx, x: int, y: int) -> bool:
    """Return True when a tile is occupied by an enemy tank.

    Args:
        ctx: Decision context with current world state.
        x: Tile X coordinate to check.
        y: Tile Y coordinate to check.

    Returns:
        True if the tile is occupied by an enemy.
    """
    return any(tank["x"] == x and tank["y"] == y for tank in ctx.filtered["tanks"].values())


def _is_occupied_by_mine(ctx: DecideCtx, x: int, y: int) -> bool:
    """Return True when a tile is occupied by a known mine.

    Args:
        ctx: Decision context with current world state.
        x: Tile X coordinate to check.
        y: Tile Y coordinate to check.

    Returns:
        True if the tile is occupied by a known mine.
    """
    return f"{x},{y}" in ctx.world["mines"]


def _pickup_target_is_blocked(ctx: DecideCtx, x: int, y: int) -> bool:
    """Return True when a pickup target cannot be occupied safely.

    Args:
        ctx: Decision context.
        x: Target X coordinate.
        y: Target Y coordinate.

    Returns:
        True if the target is blocked by an enemy or mine.
    """
    if _is_occupied_by_enemy(ctx, x, y):
        emit_ai("pickup target (%d,%d) is occupied by enemy", x, y)
        return True
    if _is_occupied_by_mine(ctx, x, y):
        emit_ai("pickup target (%d,%d) is occupied by mine", x, y)
        return True
    return False


def _walk_or_teleport_with_terrain(
    ctx: DecideCtx,
    tx: int,
    ty: int,
    sx: int,
    sy: int,
    *,
    pickup_kind: str | None,
) -> BotCommand | None:
    """Resolve movement when terrain/pathfinding is available."""
    terrain = ctx.terrain
    assert terrain is not None  # caller guarantees this
    blocked_coords = ctx.world["mines"].keys()
    if pickup_kind is not None and _pickup_target_is_blocked(ctx, tx, ty):
        return None
    if pickup_kind is not None and abs(sx - tx) <= 1 and abs(sy - ty) <= 1:
        emit_ai("adjacent to pickup target at (%d,%d)", tx, ty)
        return _make_pickup_command(pickup_kind, tx, ty)
    if not is_pickup_target_actionable(ctx, tx, ty):
        return _approach_command(ctx, tx, ty, pickup_kind=pickup_kind)
    if is_direct_path_clear(terrain, sx, sy, tx, ty, blocked_coords):
        return _direct_move_command(ctx, tx, ty, pickup_kind=pickup_kind)
    left, top, right, bottom = local_actionable_bounds(ctx)
    waypoint = find_path_segment_target(
        terrain,
        sx,
        sy,
        tx,
        ty,
        blocked_coords,
        min_x=left,
        min_y=top,
        max_x=right,
        max_y=bottom,
    )
    if waypoint is not None:
        move_cmd = _waypoint_move_command(ctx, tx, ty, waypoint)
        if move_cmd is not None:
            return move_cmd
    return _teleport_fallback_command(terrain, sx, sy, tx, ty, ctx.world["mines"])


def _walk_or_teleport_without_terrain(
    ctx: DecideCtx,
    tx: int,
    ty: int,
    *,
    pickup_kind: str | None,
) -> BotCommand | None:
    """Resolve movement when only local occupancy checks are available."""
    if not is_pickup_target_actionable(ctx, tx, ty):
        return _approach_command(ctx, tx, ty, pickup_kind=pickup_kind)
    if pickup_kind is not None:
        if _pickup_target_is_blocked(ctx, tx, ty):
            return None
        return _make_pickup_command(pickup_kind, tx, ty)
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
    *,
    pickup_kind: str | None,
) -> BotCommand | None:
    """Return a non-pickup approach command for an off-viewport target."""
    approach_x, approach_y = _approach_target(ctx, tx, ty)
    target_kind = "pickup" if pickup_kind is not None else "move"
    emit_ai(
        "%s target (%d,%d) is outside viewport, approaching via (%d,%d)",
        target_kind,
        tx,
        ty,
        approach_x,
        approach_y,
    )
    return walk_or_teleport(ctx, approach_x, approach_y, pickup_kind=None)


def _direct_move_command(
    ctx: DecideCtx,
    tx: int,
    ty: int,
    *,
    pickup_kind: str | None,
) -> BotCommand | None:
    """Return a direct move/pickup command when the straight path is clear."""
    if not is_pickup_target_actionable(ctx, tx, ty):
        emit_ai("direct target (%d,%d) is outside viewport", tx, ty)
        return None
    if pickup_kind is not None:
        if _pickup_target_is_blocked(ctx, tx, ty):
            return None
        return _make_pickup_command(pickup_kind, tx, ty)
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


def _waypoint_move_command(
    ctx: DecideCtx,
    tx: int,
    ty: int,
    waypoint: tuple[int, int],
) -> BotCommand | None:
    """Return an A*-derived waypoint move when the waypoint is usable."""
    left, top, right, bottom = local_actionable_bounds(ctx)
    wx, wy = waypoint
    if not (left <= wx <= right and top <= wy <= bottom):
        emit_ai(
            "waypoint (%d,%d) for (%d,%d) is outside viewport (%d,%d)-(%d,%d)",
            wx,
            wy,
            tx,
            ty,
            left,
            top,
            right,
            bottom,
        )
        return None
    sx, sy = ctx.self_state["x"], ctx.self_state["y"]
    if wx == sx and wy == sy:
        emit_ai("waypoint is self position, skipping")
        return None
    if is_move_target_failed(wx, wy, ctx.timestamp_ms):
        emit_ai("waypoint (%d,%d) recently failed, skipping", wx, wy)
        return None
    if _is_occupied_by_enemy(ctx, wx, wy):
        emit_ai("waypoint (%d,%d) is occupied by enemy", wx, wy)
        return None
    if _is_occupied_by_mine(ctx, wx, wy):
        emit_ai("waypoint (%d,%d) is occupied by mine", wx, wy)
        return None
    emit_ai("walking toward (%d,%d) via (%d,%d)", tx, ty, wx, wy)
    return make_move_command(wx, wy)


def _teleport_fallback_command(
    terrain: TerrainMapProtocol,
    sx: int,
    sy: int,
    tx: int,
    ty: int,
    blocked_mines: dict[str, MineStateDict],
) -> BotCommand | None:
    """Return a teleport command for a terrain-blocked target when possible."""
    landing = find_teleport_landing_tile(terrain, sx, sy, tx, ty, blocked_mines)
    if landing is None:
        emit_ai("blocked target at (%d,%d) has no passable landing tile", tx, ty)
        return None
    lx, ly = landing
    emit_ai("terrain blocked, teleporting near target to (%d,%d)", lx, ly)
    return make_teleport_command(lx, ly)


__all__ = [
    "is_pickup_target_actionable",
    "select_exploration_command",
    "viewport_exploration_candidates",
    "walk_or_teleport",
]
