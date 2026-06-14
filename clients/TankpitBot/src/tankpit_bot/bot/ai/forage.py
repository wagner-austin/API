"""Equipment foraging: grid-sweep the built-in radar to refill extra radars.

Equipment containers are radar-only-visible and the only source of
extra radars, so at zero extra radars the bot is in a spiral: it needs
a viewport-wide sweep to find equipment, but only equipment yields the
radars that power that sweep. A human escapes by covering ground with
the free built-in 5x5, sweeping new tiles until containers turn up.

This module is the zero-extra search leg of that escape. Because the
built-in footprint is exactly 5x5
(:data:`tankpit_bot.bot.ai.scan_coverage.FORAGE_CELL_SIZE`), the sweep
is a deterministic grid walk: stand in an uncovered cell, fire the
free radar (which marks the cell via the dispatch funnel), then walk
to the nearest still-uncovered cell. It runs only when extras are
exhausted -- where every scan is built-in anyway, so it can never
spend an extra and the death-spiral guard from run 20260611-232301
stays intact. Once even one extra is collected the equipment-recovery
owner switches to the viewport sweep instead.

WHEN the bot restocks equipment -- the radar break/resume hysteresis
and its precedence over hunting and fuel recovery -- is owned by
:mod:`tankpit_bot.bot.ai.mode_controller`; this module only decides
HOW to search the ground once restock holds the tick at zero extras.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
    can_afford_teleport_search,
    can_use_radar,
    clear_resource_target,
    make_decision,
    mark_scan_dispatched,
)
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai.scan_coverage import cell_center, is_cell_covered, local_scan_cell_key
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_radar_command
from tankpit_bot.runtime_logging import emit_ai

# Cells outside this chebyshev ring from the tank's cell are too far to
# be worth a foraging hop; if every cell within it is covered the local
# area is swept and the sweep yields to the recovery fallback.
_FORAGE_SEARCH_RING_LIMIT = 12

# Tiles 0..255 map to cells 0..51 under the 5-tile grid.
_MIN_CELL = 0
_MAX_CELL = 255 // 5


def select_forage_cell_target(ctx: DecideCtx) -> tuple[int, int] | None:
    """Return the nearest uncovered grid-cell center worth sweeping to.

    Cells are searched in expanding chebyshev rings around the tank's
    own cell, nearest first, skipping the current cell and any cell
    whose built-in coverage is still live. The chosen cell's center is
    where a built-in scan covers exactly that cell.

    Args:
        ctx: Decision context.

    Returns:
        ``(x, y)`` center tile of the nearest uncovered cell, or
        ``None`` when every cell within the search ring is covered.
    """
    self_cell_x = ctx.self_state["x"] // 5
    self_cell_y = ctx.self_state["y"] // 5
    cells = ctx.ai_state["local_scan_cells"]
    for ring in range(1, _FORAGE_SEARCH_RING_LIMIT + 1):
        candidate = _nearest_uncovered_in_ring(ctx, cells, self_cell_x, self_cell_y, ring)
        if candidate is not None:
            return candidate
    return None


def _nearest_uncovered_in_ring(
    ctx: DecideCtx,
    cells: dict[str, int],
    self_cell_x: int,
    self_cell_y: int,
    ring: int,
) -> tuple[int, int] | None:
    """Return the closest uncovered cell center on one chebyshev ring.

    Args:
        ctx: Decision context.
        cells: Coverage grid keyed by ``"cx,cy"``.
        self_cell_x: Tank's own cell X index.
        self_cell_y: Tank's own cell Y index.
        ring: Chebyshev ring distance to scan.

    Returns:
        Center tile of the nearest in-bounds uncovered cell on the
        ring, or ``None`` when the ring has none.
    """
    best: tuple[int, int] | None = None
    best_distance = 0
    for cell_x in range(self_cell_x - ring, self_cell_x + ring + 1):
        for cell_y in range(self_cell_y - ring, self_cell_y + ring + 1):
            if max(abs(cell_x - self_cell_x), abs(cell_y - self_cell_y)) != ring:
                continue
            if not (_MIN_CELL <= cell_x <= _MAX_CELL and _MIN_CELL <= cell_y <= _MAX_CELL):
                continue
            if is_cell_covered(cells, cell_x, cell_y, ctx.timestamp_ms):
                continue
            center_x, center_y = cell_center(cell_x, cell_y)
            distance = abs(center_x - ctx.self_state["x"]) + abs(center_y - ctx.self_state["y"])
            if best is None or distance < best_distance:
                best = (center_x, center_y)
                best_distance = distance
    return best


def plan_forage_search(
    ctx: DecideCtx,
    ai_state: AIStateDict,
    score: int,
) -> TickDecisionDict | None:
    """Plan the next equipment grid-sweep action.

    Scans the current cell with the free built-in radar when it has no
    live coverage, otherwise walks (or affordably hops) to the nearest
    uncovered cell center. Returns ``None`` when neither is possible so
    the caller can fall back to its existing search.

    Args:
        ctx: Decision context.
        ai_state: Base AI state to rewrite for the produced command.
        score: Behavior score for the produced decision.

    Returns:
        Foraging radar or move decision, or ``None`` when the local
        grid is fully swept or no action is affordable.
    """
    self_cell_x = ctx.self_state["x"] // 5
    self_cell_y = ctx.self_state["y"] // 5
    here_uncovered = not is_cell_covered(
        ai_state["local_scan_cells"],
        self_cell_x,
        self_cell_y,
        ctx.timestamp_ms,
    )
    if here_uncovered and can_use_radar(ctx):
        emit_ai(
            "forage radar at cell %s (free built-in, extras=0)",
            local_scan_cell_key(ctx.self_state["x"], ctx.self_state["y"]),
        )
        return make_decision(
            make_radar_command(),
            "COLLECT_EQUIPMENT",
            score,
            0,
            0,
            "forage_radar",
            mark_scan_dispatched(ctx, clear_resource_target(ai_state)),
            ctx.equip,
        )

    target = select_forage_cell_target(ctx)
    if target is None:
        return None
    target_x, target_y = target
    command = walk_or_teleport(ctx, target_x, target_y, pickup_kind=None)
    if command is None:
        return None
    if command["cmd_type"] == "teleport" and not can_afford_teleport_search(
        ctx,
        target_x,
        target_y,
    ):
        return None
    emit_ai("forage hop to uncovered cell center (%d,%d)", target_x, target_y)
    return make_decision(
        command,
        "COLLECT_EQUIPMENT",
        score,
        target_x,
        target_y,
        "forage_sweep",
        clear_resource_target(ai_state),
        ctx.equip,
    )


__all__ = [
    "plan_forage_search",
    "select_forage_cell_target",
]
