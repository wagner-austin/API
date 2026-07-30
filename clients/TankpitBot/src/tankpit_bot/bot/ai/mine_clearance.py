"""Mine-clearance shot planning for covered containers.

User doctrine ([[flag-triage-20260729]] F3, flags s1-4/s1-8, s2-6,
s3-2, s3-14; [[mine-mechanics]] § rank-dependent cascade): equipment
or fuel covered by an enemy mine field needs NO path clearing — one
clear-line shot AT the container's tile detonates the covering mine
plus every cardinally/diagonally adjacent mine at private and above
(a recruit's shot clears only the directly-hit mine), and the
follow-up teleport then lands on the exposed container and collects
("1 single shot can clear liek 9 mines. which may unlock multiple
equipment containers"). Mine shots consume NO inventory (user law
2026-07-30: "shooting a mine doesnt cost any inventory. you click
and it shoots a single shot, and destroys the mines") — the clearance
is free apart from the tick. Mines never occlude the shot line — only
mountains and movable land blocks do — so dense fields are shot
straight through ([[physics/line_of_sight|line_of_sight]] is the one
shared clearance test).

This module is the pure planner: given the world, pick the best
covered container to shoot. Dispatch (spending the dual, then the
collect teleport on the now-exposed containers) belongs to the
collect cascade's consumers.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.physics.line_of_sight import is_shot_line_clear
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

# Blast reach of one shot at a mine tile, by shooter rank
# ([[mine-mechanics]] [^8]): recruit (rank 0) destroys only the
# directly-hit mine; private and above (rank >= 1) destroy the target
# mine plus all 8 cardinal/diagonal neighbors.
_RECRUIT_RANK = 0


def _blast_tiles(center_x: int, center_y: int, rank: int) -> list[tuple[int, int]]:
    """Return the tiles a shot at the center clears mines from.

    Args:
        center_x: Aim tile X.
        center_y: Aim tile Y.
        rank: Shooter's true rank, ``0`` (recruit) through ``8``.

    Returns:
        The center tile alone at recruit; the full 3x3 at private+.
    """
    if rank == _RECRUIT_RANK:
        return [(center_x, center_y)]
    return [
        (center_x + dx, center_y + dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if 0 <= center_x + dx <= 255 and 0 <= center_y + dy <= 255
    ]


def find_mine_clearance_shot(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None,
) -> tuple[int, int] | None:
    """Pick the covered container whose clearance shot exposes the most.

    A candidate aim is any tracked container inside the visible
    viewport whose own tile carries a hostile mine and whose straight
    shot line from the bot is clear (rock and movable land blocks
    occlude; water, mines, tanks, and containers never do). Candidates
    are scored by how many mine-covered containers the shot's blast
    area would expose — one dual can unlock several pickups at once —
    with ties broken toward the nearest aim.

    Args:
        world: Current world state.
        self_state: The bot's own state (position and rank).
        terrain: Static field-image map; ``None`` trusts wire patches
            alone.

    Returns:
        The best ``(x, y)`` aim tile, or ``None`` when no covered
        container in view has a clear shot line.
    """
    mines = hostile_mines(world)
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    covered: list[tuple[int, int]] = []
    for container_key, container in world["containers"].items():
        if container_key not in mines:
            continue
        if not (left <= container["x"] <= right and top <= container["y"] <= bottom):
            continue
        covered.append((container["x"], container["y"]))
    if not covered:
        return None
    covered_set = set(covered)
    best: tuple[int, int] | None = None
    best_key: tuple[int, int] | None = None
    for aim_x, aim_y in covered:
        if not is_shot_line_clear(
            self_state["x"],
            self_state["y"],
            aim_x,
            aim_y,
            terrain,
            world["terrain"],
        ):
            continue
        blast = _blast_tiles(aim_x, aim_y, self_state["rank"])
        exposed = sum(1 for tile in blast if tile in covered_set)
        distance = abs(self_state["x"] - aim_x) + abs(self_state["y"] - aim_y)
        score_key = (-exposed, distance)
        if best_key is None or score_key < best_key:
            best, best_key = (aim_x, aim_y), score_key
    return best


__all__ = [
    "find_mine_clearance_shot",
]
