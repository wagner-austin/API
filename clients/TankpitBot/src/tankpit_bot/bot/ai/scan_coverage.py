"""Built-in-radar coverage grid primitives for equipment foraging.

The built-in radar reveals an exact 5x5 block (chebyshev radius 2,
wire-verified 2026-06-12) centered on the tank, versus the extra
radar's full viewport. Unlike the extra radar, a built-in scan marks
no coverage anywhere in world state, so a bot at zero extra radars has
no record of which ground it has already swept and re-scans the same
tiles forever.

This module is the coverage memory: a grid of 5x5 cells keyed by cell
index, each carrying the timestamp of the most recent scan centered in
it. It is a pure leaf -- no AI or world-state imports -- so the
dispatch funnel (:func:`tankpit_bot.bot.ai.context.mark_scan_dispatched`)
can record coverage and the forager can read it without an import
cycle.

The grid is the bot's search belief, not server truth, so it lives in
``AIStateDict.local_scan_cells``.
"""

from __future__ import annotations

# The built-in radar diameter. Cells this wide tile the map into
# adjacent, non-overlapping blocks: a scan centered in a cell covers
# exactly that cell, so stepping cell-to-cell gives complete coverage
# with no wasted re-scans.
FORAGE_CELL_SIZE = 5

# A swept cell is re-foraged after this interval -- long enough to push
# the grid sweep across the map before doubling back, short enough that
# equipment appearing later is eventually re-discovered.
FORAGE_COVERAGE_TTL_MS = 180000


def local_scan_cell_key(x: int, y: int) -> str:
    """Return the coverage-grid cell key for a tile.

    Args:
        x: Tile X coordinate.
        y: Tile Y coordinate.

    Returns:
        ``"cx,cy"`` cell-index key for the 5x5 block containing the
        tile.
    """
    return f"{x // FORAGE_CELL_SIZE},{y // FORAGE_CELL_SIZE}"


def cell_center(cell_x: int, cell_y: int) -> tuple[int, int]:
    """Return the center tile of a coverage-grid cell.

    A scan centered here covers exactly the cell's 5x5 block, so the
    forager always navigates to cell centers.

    Args:
        cell_x: Cell X index.
        cell_y: Cell Y index.

    Returns:
        ``(x, y)`` center tile of the cell.
    """
    half = FORAGE_CELL_SIZE // 2
    return (cell_x * FORAGE_CELL_SIZE + half, cell_y * FORAGE_CELL_SIZE + half)


def is_cell_covered(
    local_scan_cells: dict[str, int],
    cell_x: int,
    cell_y: int,
    now_ms: int,
) -> bool:
    """Return True when a cell carries a live built-in-scan mark.

    Args:
        local_scan_cells: Coverage grid keyed by ``"cx,cy"`` with
            scan timestamps.
        cell_x: Cell X index.
        cell_y: Cell Y index.
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        True if the cell was scanned within
        :data:`FORAGE_COVERAGE_TTL_MS`.
    """
    scanned_ms = local_scan_cells.get(f"{cell_x},{cell_y}")
    return scanned_ms is not None and now_ms - scanned_ms <= FORAGE_COVERAGE_TTL_MS


def record_local_scan(
    local_scan_cells: dict[str, int],
    x: int,
    y: int,
    now_ms: int,
) -> dict[str, int]:
    """Return the coverage grid with the scan's cell marked and pruned.

    Records the cell containing ``(x, y)`` -- the tile the radar was
    centered on -- and drops cells whose marks have aged past the
    coverage TTL so the grid cannot grow without bound across a long
    session.

    Args:
        local_scan_cells: Existing coverage grid.
        x: Scan-center X coordinate.
        y: Scan-center Y coordinate.
        now_ms: Scan timestamp recorded for the cell.

    Returns:
        New coverage grid mapping.
    """
    pruned = {
        key: scanned_ms
        for key, scanned_ms in local_scan_cells.items()
        if now_ms - scanned_ms <= FORAGE_COVERAGE_TTL_MS
    }
    pruned[local_scan_cell_key(x, y)] = now_ms
    return pruned


__all__ = [
    "FORAGE_CELL_SIZE",
    "FORAGE_COVERAGE_TTL_MS",
    "cell_center",
    "is_cell_covered",
    "local_scan_cell_key",
    "record_local_scan",
]
