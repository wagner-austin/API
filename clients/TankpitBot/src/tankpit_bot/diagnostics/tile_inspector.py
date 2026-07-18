"""Inspect terrain at a coordinate against the production passability logic.

Use cases:

* Confirm whether a fuel target rejected as ``blocked_no_landing`` is
  genuinely surrounded by water/rocks or whether the rejection logic is
  over-eager.
* Audit candidate teleport targets before committing them to the AI
  planner.
* Compare the bot's view of a tile against the game's actual field
  bitmap.

The inspector loads the requested field GIF, builds a
:class:`tankpit_bot.terrain.TerrainMap`, queries the 3x3 neighborhood
around the target, and runs the production
:func:`tankpit_bot.bot.ai.equipment.find_teleport_landing_tile` /
:func:`tankpit_bot.bot.ai.equipment.is_reachable` against it. Output is
a strict :class:`TileInspectionDict` that is round-trippable through
:func:`encode_tile_inspection` and renderable as a terminal report via
:func:`render_tile_inspection`.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.bot.ai.equipment_search import find_teleport_landing_tile, is_reachable
from tankpit_bot.diagnostics.tile_inspector_types import (
    NeighborTileDict,
    TileInspectionDict,
)
from tankpit_bot.terrain import TerrainMap

log = get_logger(__name__)


_COMPASS_OFFSETS: tuple[tuple[str, int, int], ...] = (
    ("N", 0, -1),
    ("NE", 1, -1),
    ("E", 1, 0),
    ("SE", 1, 1),
    ("S", 0, 1),
    ("SW", -1, 1),
    ("W", -1, 0),
    ("NW", -1, -1),
)


_CARDINAL_OFFSETS: tuple[tuple[str, int, int], ...] = (
    ("N", 0, -1),
    ("E", 1, 0),
    ("S", 0, 1),
    ("W", -1, 0),
)


_MAP_MIN: int = 0
_MAP_MAX: int = 255


def _is_in_bounds(x: int, y: int) -> bool:
    """Return True when ``(x, y)`` is within the 256x256 game grid."""
    return _MAP_MIN <= x <= _MAP_MAX and _MAP_MIN <= y <= _MAP_MAX


def _build_neighbor(
    terrain: TerrainMap,
    direction: str,
    target_x: int,
    target_y: int,
    dx: int,
    dy: int,
) -> NeighborTileDict:
    """Build one :class:`NeighborTileDict` for the compass offset ``(dx, dy)``."""
    nx = target_x + dx
    ny = target_y + dy
    in_bounds = _is_in_bounds(nx, ny)
    if in_bounds:
        terrain_char = terrain.get_terrain(nx, ny)
        passable = terrain.is_passable(nx, ny)
    else:
        terrain_char = " "
        passable = False
    return NeighborTileDict(
        direction=direction,
        x=nx,
        y=ny,
        terrain=terrain_char,
        passable=passable,
        in_bounds=in_bounds,
    )


def _resolve_landing_choice(
    terrain: TerrainMap,
    target_x: int,
    target_y: int,
) -> str:
    """Return a human-readable label describing why
    :func:`find_teleport_landing_tile` chose its landing tile.

    The production function walks the cardinal directions when the
    target tile itself is not passable; this helper reproduces the same
    walk and reports which direction won so the rendered report can
    explain the choice.
    """
    if terrain.is_passable(target_x, target_y):
        return "target_is_passable"
    for direction, dx, dy in _CARDINAL_OFFSETS:
        nx = target_x + dx
        ny = target_y + dy
        if not _is_in_bounds(nx, ny):
            continue
        if terrain.is_passable(nx, ny):
            return f"adjacent:{direction}"
    return "no_landing_found"


def inspect_tile(
    field_gif_path: Path,
    *,
    target_x: int,
    target_y: int,
    from_x: int,
    from_y: int,
) -> TileInspectionDict:
    """Build a :class:`TileInspectionDict` for one coordinate.

    Args:
        field_gif_path: Local path to the field minimap GIF (e.g.
            ``field01_r.gif``). The same image the bot loads at runtime.
        target_x: Coordinate to inspect.
        target_y: Coordinate to inspect.
        from_x: Origin X used for the reachability check. Pass ``-1``
            to skip the reachability calculation (the report's
            ``reachable`` field is then ``False``).
        from_y: Origin Y used for the reachability check.

    Returns:
        Fully populated :class:`TileInspectionDict`.

    Raises:
        FileNotFoundError: When ``field_gif_path`` does not exist on
            disk. Failing fast keeps the inspector from rendering a
            silently-empty report.
        ValueError: When the loaded image is not 256x256 (propagated
            from :class:`TerrainMap`).
    """
    if not _test_hooks.path_exists(field_gif_path):
        raise FileNotFoundError(f"field GIF not found: {field_gif_path}")
    terrain = TerrainMap(field_gif_path)
    target_in_bounds = _is_in_bounds(target_x, target_y)
    target_terrain = terrain.get_terrain(target_x, target_y) if target_in_bounds else " "
    target_passable = terrain.is_passable(target_x, target_y) if target_in_bounds else False
    neighbors = [
        _build_neighbor(terrain, direction, target_x, target_y, dx, dy)
        for direction, dx, dy in _COMPASS_OFFSETS
    ]
    landing_resolution = _resolve_landing_choice(terrain, target_x, target_y)
    landing = find_teleport_landing_tile(
        terrain,
        target_x,
        target_y,
    )
    landing_tile_x = landing[0] if landing is not None else -1
    landing_tile_y = landing[1] if landing is not None else -1
    if from_x >= 0 and from_y >= 0 and _is_in_bounds(from_x, from_y) and target_in_bounds:
        reachable = is_reachable(terrain, from_x, from_y, target_x, target_y)
    else:
        reachable = False
    return TileInspectionDict(
        field_image=field_gif_path.name,
        target_x=target_x,
        target_y=target_y,
        target_terrain=target_terrain,
        target_passable=target_passable,
        target_in_bounds=target_in_bounds,
        neighbors=neighbors,
        landing_tile_x=landing_tile_x,
        landing_tile_y=landing_tile_y,
        landing_resolution=landing_resolution,
        from_x=from_x,
        from_y=from_y,
        reachable=reachable,
    )


def render_tile_inspection(report: TileInspectionDict) -> str:
    """Render a :class:`TileInspectionDict` to a human-readable string."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("TANKPIT TILE INSPECTION")
    lines.append("=" * 72)
    lines.append(f"Field image: {report['field_image']}")
    lines.append(f"Target: ({report['target_x']}, {report['target_y']})")
    lines.append(
        f"Target terrain: {report['target_terrain']!r} "
        f"passable={report['target_passable']} in_bounds={report['target_in_bounds']}"
    )
    lines.append("")
    lines.append("=== NEIGHBORS (8-tile ring around target) ===")
    for neighbor in report["neighbors"]:
        marker = "[OK]" if neighbor["passable"] else "[X ]"
        lines.append(
            f"  {marker} {neighbor['direction']:2s} ({neighbor['x']:3d},{neighbor['y']:3d}) "
            f"terrain={neighbor['terrain']!r} passable={neighbor['passable']} "
            f"in_bounds={neighbor['in_bounds']}"
        )
    lines.append("")
    landing_line = (
        f"Landing tile: ({report['landing_tile_x']},{report['landing_tile_y']})"
        if report["landing_tile_x"] >= 0
        else "Landing tile: NONE (target unreachable as a teleport destination)"
    )
    lines.append("=== TELEPORT LANDING RESOLUTION ===")
    lines.append(f"  {landing_line}")
    lines.append(f"  Resolution path: {report['landing_resolution']}")
    lines.append("")
    if report["from_x"] >= 0 and report["from_y"] >= 0:
        lines.append("=== REACHABILITY (A* from origin) ===")
        lines.append(
            f"  from=({report['from_x']},{report['from_y']}) "
            f"to=({report['target_x']},{report['target_y']}) "
            f"reachable={report['reachable']}"
        )
    else:
        lines.append("=== REACHABILITY ===")
        lines.append("  (no origin provided; pass --from x,y to enable)")
    lines.append("=" * 72)
    return "\n".join(lines)


def _parse_coord_pair(raw: str, *, flag: str) -> tuple[int, int]:
    """Parse an ``x,y`` argument into a tuple of ints.

    Args:
        raw: Comma-separated coordinate string.
        flag: CLI flag name for error messages.

    Returns:
        ``(x, y)`` tuple.

    Raises:
        ValueError: When the input is not in ``int,int`` form.
    """
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError(f"{flag} expects two comma-separated ints, got {raw!r}")
    return (int(parts[0]), int(parts[1]))


def _parse_optional_flag(argv: list[str], flag: str) -> str | None:
    """Return the value for ``flag`` in ``argv``, or ``None`` if absent."""
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"{flag} requires a value")
    return argv[index + 1]


def _require_positional(argv: list[str], position: int, name: str) -> str:
    """Return the ``position``-th positional argument or raise."""
    positional = [arg for arg in argv if not arg.startswith("--")]
    if position >= len(positional):
        raise ValueError(f"missing required positional argument {name!r}")
    return positional[position]


def main() -> int:
    """Run the ``tankpit-tile-info`` CLI entrypoint.

    Usage::

        tankpit-tile-info <field-gif> <x> <y> [--from sx,sy]

    Returns:
        Process exit code (``0`` on success).

    Raises:
        ValueError: When required arguments are missing or malformed.
        FileNotFoundError: When the field GIF does not exist.
    """
    setup_rich_logging(level="INFO")
    full_argv = list(_test_hooks.get_argv())
    user_args = full_argv[1:] if full_argv else []
    field_gif = Path(_require_positional(user_args, 0, "field-gif"))
    target_x = int(_require_positional(user_args, 1, "x"))
    target_y = int(_require_positional(user_args, 2, "y"))
    from_raw = _parse_optional_flag(user_args, "--from")
    if from_raw is None:
        from_x = -1
        from_y = -1
    else:
        from_x, from_y = _parse_coord_pair(from_raw, flag="--from")
    report = inspect_tile(
        field_gif,
        target_x=target_x,
        target_y=target_y,
        from_x=from_x,
        from_y=from_y,
    )
    log.info("%s", render_tile_inspection(report))
    return 0


__all__ = [
    "inspect_tile",
    "main",
    "render_tile_inspection",
]
