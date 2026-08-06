"""Mine-clearance shot planning: one free shot opens mined access.

User doctrine ([[flag-triage-20260729]] F3, flags s1-4/s1-8, s2-6,
s3-2, s3-14; [[mine-mechanics]] § rank-dependent cascade): a
clear-line shot AT a mine detonates it plus every cardinally or
diagonally adjacent mine at private and above (a recruit's shot
clears only the directly-hit mine). Mine shots consume NO inventory
(user law 2026-07-30: "shooting a mine doesnt cost any inventory.
you click and it shoots a single shot, and destroys the mines") —
the clearance is free apart from the tick. Mines never occlude the
shot line — only mountains and movable land blocks do — so dense
fields are shot straight through
([[physics/line_of_sight|line_of_sight]] is the one shared test).

The trigger is the GENERAL condition — a known hostile mine stands
between the bot and a worthwhile container's service tiles — not an
enumeration of special cases. Session bot-20260805-173034 proved why:
equipment on water with mined flanks fit neither of the old two
triggers (mine ON the container tile; mine on a planned walk
corridor), so the bot re-aimed 1,068 displaced teleports at the mine
for 43 minutes with the free unlock shot sitting unwired. A shot is
planned whenever it provably restores access: it exposes a covered
container's own tile, or its blast opens an attainable teleport
landing ([[teleport-mechanics]] displacement law) that no service
tile offers today.

This module is the pure planner: given the world, pick the best aim.
Dispatch (spending the free single, then the collect teleport on the
now-open access) belongs to the collect cascade's consumers.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.ai.reachability import find_attainable_landing_tile
from tankpit_bot.physics.line_of_sight import is_shot_line_clear, shot_line_tiles
from tankpit_bot.state import SelfStateDict, WorldStateDict
from tankpit_bot.state.types import ContainerStateDict, MineStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

# Blast reach of one shot at a mine tile, by shooter rank
# ([[mine-mechanics]] [^8]): recruit (rank 0) destroys only the
# directly-hit mine; private and above (rank >= 1) destroy the target
# mine plus all 8 cardinal/diagonal neighbors.
_RECRUIT_RANK = 0

# A clearance shot must expose something worth the tick it costs.
# Equipment is always worth it (four pieces per pickup, no cap
# clamp); a fuel container has to hold a real drink -- flag 8 of run
# bot-20260730-015x spent a shot un-covering a 21-volume dreg ("i
# could understand if it was a high value container... but 21 value
# fuel container?").
_MIN_CLEARANCE_FUEL_VOLUME = 100


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


def _service_tiles(goal_x: int, goal_y: int) -> list[tuple[int, int]]:
    """Return the tiles a pickup at the goal can be served from.

    The measured transfer choreography ([[fuel-system]],
    [[equipment-system]]): the tank must stand ON the container tile
    or CARDINALLY adjacent. This set is therefore also exactly the
    landing-candidate set the hop selectors scan.

    Args:
        goal_x: Container X coordinate.
        goal_y: Container Y coordinate.

    Returns:
        In-bounds goal tile plus cardinal neighbors.
    """
    tiles = [(goal_x, goal_y)]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = goal_x + dx, goal_y + dy
        if 0 <= nx <= 255 and 0 <= ny <= 255:
            tiles.append((nx, ny))
    return tiles


def _mines_after_blast(
    mines: dict[str, MineStateDict],
    aim_x: int,
    aim_y: int,
    rank: int,
) -> dict[str, MineStateDict]:
    """Return the mine layer as it would stand after a shot at the aim.

    The 0x45 detonation removes every mine in the blast area
    regardless of team ([[mine-mechanics]] cascade dispatch law).

    Args:
        mines: Full known mine layer indexed by ``"x,y"``.
        aim_x: Aim tile X.
        aim_y: Aim tile Y.
        rank: Shooter's true rank (blast reach).

    Returns:
        The mine layer minus every mine inside the blast.
    """
    blast = set(_blast_tiles(aim_x, aim_y, rank))
    return {key: mine for key, mine in mines.items() if (mine["x"], mine["y"]) not in blast}


def _aim_opens_target(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None,
    aim_x: int,
    aim_y: int,
    container: ContainerStateDict,
    *,
    covered: bool,
    blocked: bool,
) -> bool:
    """Return whether a shot at the aim restores access to the container.

    Args:
        world: Current world state.
        self_state: The bot's own state (rank for blast reach).
        terrain: Composed decision terrain; ``None`` disables the
            landing-attainability arm.
        aim_x: Candidate aim X.
        aim_y: Candidate aim Y.
        container: The mined-access container being evaluated.
        covered: Container's own tile carries a hostile mine.
        blocked: No service tile offers an attainable landing today.

    Returns:
        True when the blast exposes the covered tile or opens an
        attainable landing that did not exist before the shot.
    """
    blast = set(_blast_tiles(aim_x, aim_y, self_state["rank"]))
    if covered and (container["x"], container["y"]) in blast:
        return True
    if blocked and terrain is not None:
        remaining = _mines_after_blast(world["mines"], aim_x, aim_y, self_state["rank"])
        return (
            find_attainable_landing_tile(terrain, remaining, container["x"], container["y"])
            is not None
        )
    return False


def find_service_clearance_aim(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None,
    goal_x: int,
    goal_y: int,
) -> tuple[int, int] | None:
    """Find the shot that opens teleport access to ONE specific target.

    The single-target arm of the general trigger, consumed by the
    lock-release verdict: a container with no attainable landing is
    still SERVABLE while a known hostile mine on one of its service
    tiles can be shot from here — the clearance step fires before the
    hop lanes, so the lock must hold through it rather than release
    ``unservable``.

    Args:
        world: Current world state.
        self_state: The bot's own state.
        terrain: Composed decision terrain; ``None`` cannot answer
            attainability, so no aim is proposed.
        goal_x: Target container X.
        goal_y: Target container Y.

    Returns:
        The nearest service-tile mine whose blast provably opens an
        attainable landing, or ``None`` when access is already open,
        terrain is unknown, or no such shot exists.
    """
    if terrain is None:
        return None
    if find_attainable_landing_tile(terrain, world["mines"], goal_x, goal_y) is not None:
        return None
    hostile = hostile_mines(world)
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    best: tuple[int, int] | None = None
    best_distance = 0
    for tile_x, tile_y in _service_tiles(goal_x, goal_y):
        if f"{tile_x},{tile_y}" not in hostile:
            continue
        if not (left <= tile_x <= right and top <= tile_y <= bottom):
            continue
        if not is_shot_line_clear(
            self_state["x"],
            self_state["y"],
            tile_x,
            tile_y,
            terrain,
            world["terrain"],
        ):
            continue
        remaining = _mines_after_blast(world["mines"], tile_x, tile_y, self_state["rank"])
        if find_attainable_landing_tile(terrain, remaining, goal_x, goal_y) is None:
            continue
        distance = abs(self_state["x"] - tile_x) + abs(self_state["y"] - tile_y)
        if best is None or distance < best_distance:
            best, best_distance = (tile_x, tile_y), distance
    return best


def _denied_containers(
    world: WorldStateDict,
    terrain: TerrainMapProtocol | None,
    hostile: dict[str, MineStateDict],
) -> tuple[list[tuple[ContainerStateDict, bool, bool]], set[tuple[int, int]]]:
    """Collect mine-denied containers in view and their candidate aims.

    Args:
        world: Current world state.
        terrain: Composed decision terrain; ``None`` disables the
            blocked-landing arm.
        hostile: Hostile mines indexed by ``"x,y"``.

    Returns:
        The (container, covered, blocked) triples for every worthwhile
        in-viewport container whose access is mine-denied, and the set
        of in-viewport hostile-mine service tiles that could be aimed
        at to reopen them.
    """
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    denied: list[tuple[ContainerStateDict, bool, bool]] = []
    aim_candidates: set[tuple[int, int]] = set()
    for container_key, container in world["containers"].items():
        if not (left <= container["x"] <= right and top <= container["y"] <= bottom):
            continue
        if container["is_fuel"] and container["volume"] < _MIN_CLEARANCE_FUEL_VOLUME:
            continue
        covered = container_key in hostile
        blocked = terrain is not None and (
            find_attainable_landing_tile(terrain, world["mines"], container["x"], container["y"])
            is None
        )
        if not covered and not blocked:
            continue
        denied.append((container, covered, blocked))
        for tile_x, tile_y in _service_tiles(container["x"], container["y"]):
            if f"{tile_x},{tile_y}" not in hostile:
                continue
            if left <= tile_x <= right and top <= tile_y <= bottom:
                aim_candidates.add((tile_x, tile_y))
    return denied, aim_candidates


def find_mine_clearance_shot(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None,
) -> tuple[int, int] | None:
    """Pick the mine whose clearance shot restores the most access.

    The general trigger: a worthwhile tracked container in the visible
    viewport whose access is mine-denied — its own tile carries a
    hostile mine (covered), or no service tile offers an attainable
    teleport landing (blocked, the displacement law). Candidate aims
    are the hostile mines on those containers' service tiles with a
    clear straight shot line from the bot (rock and movable land
    blocks occlude; water, mines, tanks, and containers never do).
    Aims are scored by how many denied containers the blast provably
    reopens — one free single can unlock several pickups at once —
    with ties broken toward the nearest aim.

    Args:
        world: Current world state.
        self_state: The bot's own state (position and rank).
        terrain: Composed decision terrain; ``None`` trusts wire
            patches alone and disables the blocked-landing arm.

    Returns:
        The best ``(x, y)`` aim tile, or ``None`` when no denied
        container in view has a shot that reopens it.
    """
    hostile = hostile_mines(world)
    if not hostile:
        return None
    denied, aim_candidates = _denied_containers(world, terrain, hostile)
    if not denied:
        return None
    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int, int] | None = None
    for aim_x, aim_y in aim_candidates:
        if not is_shot_line_clear(
            self_state["x"],
            self_state["y"],
            aim_x,
            aim_y,
            terrain,
            world["terrain"],
        ):
            continue
        exposed = sum(
            1
            for container, covered, blocked in denied
            if _aim_opens_target(
                world,
                self_state,
                terrain,
                aim_x,
                aim_y,
                container,
                covered=covered,
                blocked=blocked,
            )
        )
        if exposed == 0:
            continue
        distance = abs(self_state["x"] - aim_x) + abs(self_state["y"] - aim_y)
        score_key = (-exposed, distance, aim_x, aim_y)
        if best_key is None or score_key < best_key:
            best, best_key = (aim_x, aim_y), score_key
    return best


def find_corridor_clearance_shot(
    world: WorldStateDict,
    self_state: SelfStateDict,
    terrain: TerrainMapProtocol | None,
    dest_x: int,
    dest_y: int,
) -> tuple[int, int] | None:
    """Pick the first known hostile mine on the walk corridor to shoot.

    Flags s6-8/9 (run bot-20260730-021x): Yuppler laid mines in view
    and Artax walked into six of them at 45 fuel each — every hit
    arrested the move and the next tick re-dispatched the same walk
    into the next mine. The mines were KNOWN (0x4B placements land in
    the mine layer without a re-scan); the walk dispatch simply never
    consulted them. Before walking, the straight corridor to the
    destination (endpoints included) is checked for hostile mines; the
    first one with a clear shot line becomes a free clearance single
    (rank-dependent blast, [[mine-mechanics]]), and the walk proceeds
    next tick through drained ground.

    Args:
        world: Current world state.
        self_state: The bot's own state.
        terrain: Static field-image map; ``None`` trusts wire patches
            alone.
        dest_x: Walk destination X.
        dest_y: Walk destination Y.

    Returns:
        The first corridor mine's ``(x, y)`` with a clear shot line,
        or ``None`` when the corridor is mine-free or nothing on it
        can be shot from here.
    """
    mines = hostile_mines(world)
    if not mines:
        return None
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    corridor = shot_line_tiles(self_state["x"], self_state["y"], dest_x, dest_y)
    corridor.append((dest_x, dest_y))
    for tile_x, tile_y in corridor:
        if f"{tile_x},{tile_y}" not in mines:
            continue
        if not (left <= tile_x <= right and top <= tile_y <= bottom):
            continue
        if is_shot_line_clear(
            self_state["x"],
            self_state["y"],
            tile_x,
            tile_y,
            terrain,
            world["terrain"],
        ):
            return (tile_x, tile_y)
    return None


__all__ = [
    "find_corridor_clearance_shot",
    "find_mine_clearance_shot",
    "find_service_clearance_aim",
]
