"""Static world population + mined practice-room layouts.

The real field carries far more fuel than its map shows and nothing
observably spawns ([[game-economy]] 2026-07-25: the 2026-07-22
"respawn law" was falsified — all 605 "spawns" were exposures of
pre-existing ≥500-volume containers). This module seeds the sim's
static population accordingly:

* **Dotted fuel** — the standing map atlas (~620 dots live). Dots
  are exposure memory: only ~40% still hold usable fuel, so 3 of
  every 5 seeded dot containers are drained to volume 0 (they answer
  pickups with 0x52 code 4, exactly like live).
* **Hidden fuel** — invisible until the client radars its tile.
  Population, drained fraction, and small/large mix are MEASURED
  (density probe 2026-07-25, 8 extra-radar sweeps of fresh
  map-spread viewports): ~840 hidden fuel map-wide, half drained,
  2-in-5 of the stocked below 500; large volumes cycle through the
  archive band mix.
* **Hidden equipment** — same placement model; ~180 map-wide,
  measured by the same probe.

Practice layouts are REAL practice-room states lifted from archive
captures (`analysis_scripts/mine_practice_roster.py`): the full
36-bot roster (ids 500-535, 9 per team, ranks 0-1) at its actually
observed positions, plus the client's real join spawn. Layout choice
is stamp-derived and deterministic.
"""

from __future__ import annotations

import zlib
from typing import TypedDict

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.sim.spawn import find_open_tile_near
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimEquipmentDict,
    SimWorldDict,
    make_sim_tank,
    place_mine,
)

DOTTED_FUEL_COUNT = 620
"""Seeded map-atlas dots — the measured live census (569-656, ~619)."""

DOTTED_STOCKED_PERIOD = 5
"""Every 5 seeded dots, 2 hold fuel and 3 are drained (~40% hold)."""

HIDDEN_FUEL_COUNT = 840
"""Hidden fuel containers, MEASURED 2026-07-25 (density probe run 5,
`runs/probe/density-20260725-171318`): 23 hidden fuel reveals over
1,792 fresh tiles across 7 map-spread viewports ≈ 0.0128/tile ≈ 840
map-wide. Roughly half are drained (12 of 23 held fuel) — see
:data:`HIDDEN_DRAINED_PERIOD`."""

HIDDEN_DRAINED_PERIOD = 2
"""Every 2nd hidden fuel container is drained to volume 0 (measured:
12 stocked of 23 hidden reveals)."""

HIDDEN_EQUIPMENT_COUNT = 180
"""Hidden equipment containers, MEASURED same probe: 5 reveals over
1,792 fresh tiles ≈ 0.0028/tile ≈ 180 map-wide."""

MINE_DENSITY = 0.14
"""Fraction of passable tiles carrying a mine.

The archive gives TWO densities, and they measure different ground:

* The client's OPENING 0x5A patch holds 88-159 mines of its 324 tiles
  (34 sessions, median 130) — a 0.27-0.49 carpet. But that patch sits
  on the spawn, the most fought-over ground in the room.
* Across everything a session travels, the believed registry runs a
  median 383 mines in 85 components (27 sessions, replayed through
  the production dispatcher).

The sim seeds the SHAPE law uniformly at the lower figure. Carpeting
the whole room at the spawn density leaves no lanes, and the router —
which paths around revealed hostile mines exactly as the real server
does — then finds no route at all: the first cut starved a forage
session at round 11 with ``blocked_walk`` on its only container. Where
the spawn carpet ENDS is not modelled, because the archive never
watched the bot more than 32 tiles from its join and so never measured
it ([[session-state-deglobalisation]]).

The 0x4F radar delta is not an instrument for either figure: it
reports per-tile WRITES, so a session's median 12 distinct mine tiles
are the ones that CHANGED, not the ones that exist."""

MINE_SHAPE_CYCLE: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 1),
    (3, 1),
    (1, 1),
    (2, 3),
    (1, 1),
    (2, 1),
    (1, 1),
    (1, 3),
    (3, 3),
    (1, 1),
    (3, 2),
    (1, 1),
    (2, 2),
    (4, 3),
    (1, 1),
    (1, 2),
    (3, 4),
    (1, 3),
    (2, 3),
)
"""Component shapes, in the archive's measured mix.

MEASURED 2026-08-06 by replaying 27 real captures through the
PRODUCTION dispatcher and splitting the believed mine registry into
8-connected components (2,236 of them). Real minefields are not a
scatter and not a uniform grid — they are separate blobs:

* 39.3% are a SINGLE mine;
* 52.5% are press-sized, 12 tiles or fewer, with 3, 2 and 9 the
  commonest sizes (9 being a clean 3x3 press);
* 6.7% irregular clumps, 1.3% walls or lines, 0.3% blankets — the
  largest seen is 203 tiles in a 17x29 box.

Bounding boxes follow the same story: 1x1 (878), 3x3 (269), 1x2 (253),
1x3 (185), 2x3 (183), 2x2 (114). Fill ratio is a median 1.00, so the
blobs are SOLID. This cycle's twenty-one entries reproduce the size
histogram by count — eight singles, three 2-tile, three 3-tile, one
4-tile, three 6-tile, one 9-tile, two 12-tile — and leave out the rare
walls and blankets rather than place them at a rate one-in-three-
hundred cannot pin down.

Shape matters as much as count. A uniform scatter at the same density
has no lanes at all, and the sim's router — which paths around
revealed hostile mines exactly as the real server does — could then
find no route: a forage session starved at round 11 with
``blocked_walk`` on its only container, while real sessions carry a
median 383 believed mines and 85 components and play normally
([[session-state-deglobalisation]])."""

MINE_TEAM_CYCLE: tuple[int, ...] = (3, 3, 0, 3, 3, 0, 3, 0, 3, 3, 0)
"""Owning teams, in the archive's measured proportions.

The patch census counted 13,672 mined tiles for team 3 and 7,519 for
team 0 — 64.5% / 35.5%, which this eleven-slot cycle reproduces at
7/11 and 4/11. Teams 2 and 1 appeared 35 and 4 times, a combined
0.18%; placing them at all would mean inventing a rate the sample
cannot support, so the sim's field is the two colours that own it."""

LARGE_VOLUME_CYCLE: tuple[int, ...] = (
    520,
    560,
    590,
    620,
    660,
    700,
    740,
    780,
    820,
    860,
    900,
    950,
    1000,
    1050,
    1090,
    1150,
    1250,
    1400,
    505,
    980,
)
"""Deterministic ≥500 volume cycle, proportioned to the measured
large-band mix (500-509: ~1%, 510-549: ~7%, 550-599: ~8%,
600-799: ~28%, 800-1099: ~42%, 1100+: ~13% of 834 reveals)."""

SMALL_VOLUME_CYCLE: tuple[int, ...] = (34, 57, 120, 180, 250, 320, 400, 460)
"""Deterministic sub-500 volume cycle (never dots; the live off-dot
spot checks were 34 and 57)."""

SMALL_PERIOD = 5
"""2 of every 5 STOCKED hidden containers are sub-500 (density probe
2026-07-25: 5 of 12 stocked hidden reveals were below 500 — fresh
ground carries more small fuel than the archive's visited-area
reveal mix suggested)."""

_CLIENT_RANK = 1
_CLIENT_COUNTS = 25


class PracticeLayout(TypedDict):
    """One real practice-room state lifted from an archive capture."""

    provenance: str
    client_spawn: tuple[int, int]
    roster: tuple[tuple[int, int, int, int, int], ...]


PRACTICE_LAYOUTS: tuple[PracticeLayout, ...] = (
    PracticeLayout(
        provenance="bot-20260706-223721",
        client_spawn=(131, 126),
        roster=(
            (500, 0, 1, 6, 224),
            (501, 0, 1, 143, 70),
            (502, 0, 1, 1, 211),
            (503, 0, 1, 167, 221),
            (504, 0, 1, 227, 172),
            (505, 0, 0, 141, 241),
            (506, 0, 0, 97, 38),
            (507, 0, 0, 100, 229),
            (508, 0, 0, 24, 70),
            (509, 1, 1, 60, 244),
            (510, 1, 1, 23, 210),
            (511, 1, 1, 38, 1),
            (512, 1, 1, 166, 120),
            (513, 1, 1, 133, 226),
            (514, 1, 0, 52, 254),
            (515, 1, 0, 37, 232),
            (516, 1, 0, 4, 170),
            (517, 1, 0, 50, 221),
            (518, 2, 1, 211, 155),
            (519, 2, 1, 116, 56),
            (520, 2, 1, 227, 222),
            (521, 2, 1, 253, 239),
            (522, 2, 1, 219, 162),
            (523, 2, 0, 229, 80),
            (524, 2, 0, 245, 24),
            (525, 2, 0, 203, 107),
            (526, 2, 1, 216, 170),
            (527, 3, 1, 147, 220),
            (528, 3, 1, 190, 42),
            (529, 3, 1, 137, 14),
            (530, 3, 1, 122, 146),
            (531, 3, 1, 246, 245),
            (532, 3, 0, 53, 204),
            (533, 3, 0, 21, 219),
            (534, 3, 0, 46, 159),
            (535, 3, 0, 23, 226),
        ),
    ),
    PracticeLayout(
        provenance="sniff-20260720-221239",
        client_spawn=(131, 126),
        roster=(
            (500, 0, 1, 100, 193),
            (501, 0, 1, 167, 217),
            (502, 0, 1, 245, 219),
            (503, 0, 1, 163, 218),
            (504, 0, 1, 145, 254),
            (505, 0, 0, 141, 241),
            (506, 0, 0, 54, 72),
            (507, 0, 0, 100, 229),
            (508, 0, 0, 91, 41),
            (509, 1, 1, 35, 163),
            (510, 1, 1, 6, 163),
            (511, 1, 1, 23, 174),
            (512, 1, 1, 133, 6),
            (513, 1, 1, 240, 61),
            (514, 1, 0, 52, 254),
            (515, 1, 0, 23, 237),
            (516, 1, 0, 12, 218),
            (517, 1, 0, 50, 221),
            (518, 2, 1, 121, 56),
            (519, 2, 1, 2, 92),
            (520, 2, 1, 67, 208),
            (521, 2, 1, 150, 20),
            (522, 2, 1, 234, 241),
            (523, 2, 0, 138, 240),
            (524, 2, 0, 126, 138),
            (525, 2, 0, 249, 41),
            (526, 2, 0, 41, 158),
            (527, 3, 1, 147, 220),
            (528, 3, 1, 244, 194),
            (529, 3, 1, 221, 239),
            (530, 3, 1, 23, 166),
            (531, 3, 1, 246, 245),
            (532, 3, 0, 126, 226),
            (533, 3, 0, 21, 219),
            (534, 3, 0, 157, 9),
            (535, 3, 0, 23, 226),
        ),
    ),
    PracticeLayout(
        provenance="sniff-20260721-200527",
        client_spawn=(12, 56),
        roster=(
            (500, 0, 1, 220, 146),
            (501, 0, 1, 167, 217),
            (502, 0, 1, 163, 202),
            (503, 0, 1, 163, 218),
            (504, 0, 1, 145, 254),
            (505, 0, 0, 141, 241),
            (506, 0, 0, 69, 176),
            (507, 0, 0, 100, 229),
            (508, 0, 0, 133, 8),
            (509, 1, 1, 195, 225),
            (510, 1, 1, 93, 222),
            (511, 1, 1, 25, 137),
            (512, 1, 1, 28, 247),
            (513, 1, 1, 92, 214),
            (514, 1, 0, 52, 254),
            (515, 1, 0, 23, 237),
            (516, 1, 0, 192, 20),
            (517, 1, 0, 50, 221),
            (518, 2, 1, 121, 56),
            (519, 2, 1, 2, 92),
            (520, 2, 1, 67, 208),
            (521, 2, 1, 150, 20),
            (522, 2, 1, 234, 241),
            (523, 2, 0, 138, 240),
            (524, 2, 0, 126, 138),
            (525, 2, 0, 249, 41),
            (526, 2, 0, 41, 158),
            (527, 3, 1, 147, 220),
            (528, 3, 1, 244, 194),
            (529, 3, 1, 19, 38),
            (530, 3, 1, 59, 245),
            (531, 3, 1, 246, 245),
            (532, 3, 0, 126, 226),
            (533, 3, 0, 35, 235),
            (534, 3, 0, 157, 9),
            (535, 3, 0, 23, 226),
        ),
    ),
)
"""Three real layouts from three different days."""


def select_practice_layout(stamp: str) -> PracticeLayout:
    """Pick the run's layout deterministically from its stamp.

    Args:
        stamp: The run stamp (any string).

    Returns:
        One of :data:`PRACTICE_LAYOUTS` — same stamp, same layout.
    """
    return PRACTICE_LAYOUTS[zlib.crc32(stamp.encode("utf-8")) % len(PRACTICE_LAYOUTS)]


def _large_volume(index: int) -> int:
    """Return the deterministic ≥500 volume for one placement index."""
    return LARGE_VOLUME_CYCLE[index % len(LARGE_VOLUME_CYCLE)]


def _hidden_volume(index: int) -> int:
    """Return the deterministic hidden-population volume for one index.

    The measured hidden mix (density probe 2026-07-25): every 2nd
    container is drained to 0; among the stocked, 2 of every 5 draw
    from the sub-500 cycle and the rest from the large cycle.
    """
    if index % HIDDEN_DRAINED_PERIOD == HIDDEN_DRAINED_PERIOD - 1:
        return 0
    stocked_index = index // HIDDEN_DRAINED_PERIOD
    if stocked_index % SMALL_PERIOD < 2:
        return SMALL_VOLUME_CYCLE[(stocked_index // SMALL_PERIOD) % len(SMALL_VOLUME_CYCLE)]
    return _large_volume(stocked_index)


_MAP_SPAN = 256
_TILE_COUNT = _MAP_SPAN * _MAP_SPAN
_SEED_STRIDE = 97
_PROBE_STRIDE = 251

#: Components land on every third row, so a shape up to four tiles tall
#: cannot fuse with the row band above it into one blob — the merged
#: result would be the wrong SHAPE even at the right count.
_MINE_ROW_STRIDE = 3
#: One component per this many passable tiles within a seeding row.
#: With the row stride above, this lands :data:`MINE_DENSITY`.
_MINE_COMPONENT_SPACING = 9


def _next_open_tile(
    occupied: set[tuple[int, int]],
    terrain: TerrainMapProtocol,
    index: int,
) -> tuple[int, int]:
    """Return the next deterministic open tile and mark it occupied.

    The same full-cycle stride walk as :func:`sim.spawn.find_open_tile`
    but against a local occupancy set, so seeding ~2,000 entities
    stays linear instead of rescanning the world's entity lists per
    placement.

    Args:
        occupied: Tiles already taken (mutated: the pick is added).
        terrain: Static terrain of the world's field.
        index: Determinism seed for this placement.

    Returns:
        The chosen tile.

    Raises:
        RuntimeError: If the map runs out of open tiles — a seeding
            bug, never a normal outcome on field01.
    """
    start = (index * _SEED_STRIDE) % _TILE_COUNT
    for step in range(_TILE_COUNT):
        linear = (start + step * _PROBE_STRIDE) % _TILE_COUNT
        # The 0x4C skip-RLE cursor starts at (1, 1): a dot before
        # linear position 257 is unencodable on the wire, so no
        # container may sit there (any hidden container can become a
        # dot through exposure).
        if linear < _MAP_SPAN + 1:
            continue
        x, y = linear % _MAP_SPAN, linear // _MAP_SPAN
        if (x, y) in occupied or not terrain.is_passable(x, y):
            continue
        occupied.add((x, y))
        return x, y
    raise RuntimeError("seed_field_population: no open tile left on the map")


def seed_minefield(world: SimWorldDict, terrain: TerrainMapProtocol) -> int:
    """Lay the room's standing minefield across the passable map.

    Mines arrive as separate solid COMPONENTS in the measured mix of
    :data:`MINE_SHAPE_CYCLE` — mostly single mines and press-sized
    blobs — because that is what the archive shows, and because the
    gaps between them are what a route needs.

    Mines are seeded INDEPENDENTLY of containers, because the game
    lets them share a tile ([[mine-mechanics]]: "Containers can
    coexist with mines on the same tile") and those shared tiles are
    exactly the ones the bot's clearance and landing-displacement
    machinery exists for. Living tanks are skipped — a placement never
    lands under one.

    Deterministic: component origins come from a fixed raster walk, so
    the same field yields the same minefield on every run.

    Args:
        world: Simulated world (mutated: ``mines`` filled).
        terrain: Static terrain of the world's field.

    Returns:
        How many mines were laid.
    """
    occupied = {(tank["x"], tank["y"]) for tank in world["tanks"].values() if tank["alive"]}
    passable = 0
    components = 0
    for linear in range(_TILE_COUNT):
        x, y = linear % _MAP_SPAN, linear // _MAP_SPAN
        if not terrain.is_passable(x, y):
            continue
        passable += 1
        if y % _MINE_ROW_STRIDE != 0 or passable % _MINE_COMPONENT_SPACING != 0:
            continue
        width, height = MINE_SHAPE_CYCLE[components % len(MINE_SHAPE_CYCLE)]
        team = MINE_TEAM_CYCLE[components % len(MINE_TEAM_CYCLE)]
        components += 1
        for dy in range(height):
            for dx in range(width):
                tile_x, tile_y = x + dx, y + dy
                if not (0 <= tile_x < _MAP_SPAN and 0 <= tile_y < _MAP_SPAN):
                    continue
                if not terrain.is_passable(tile_x, tile_y) or (tile_x, tile_y) in occupied:
                    continue
                place_mine(world, tile_x, tile_y, team)
    return len(world["mines"])


def seed_field_population(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    *,
    seed: int,
) -> None:
    """Seed the static container population onto the world.

    Args:
        world: Simulated world (mutated: containers and equipment
            appended).
        terrain: Static terrain of the world's field.
        seed: Determinism seed (vary per layout so worlds differ).

    Raises:
        RuntimeError: If the map runs out of open tiles — a seeding
            bug, never a normal outcome on field01.
    """
    occupied: set[tuple[int, int]] = {
        (tank["x"], tank["y"]) for tank in world["tanks"].values() if tank["alive"]
    }
    occupied.update((c["x"], c["y"]) for c in world["containers"])
    occupied.update((e["x"], e["y"]) for e in world["equipment"])
    occupied.update((m["x"], m["y"]) for m in world["mines"].values())
    occupied.update((b["x"], b["y"]) for b in world["blocks"])
    index = seed
    for dot_i in range(DOTTED_FUEL_COUNT):
        x, y = _next_open_tile(occupied, terrain, index)
        stocked = dot_i % DOTTED_STOCKED_PERIOD < 2
        volume = _large_volume(dot_i) if stocked else 0
        world["containers"].append(SimContainerDict(x=x, y=y, volume=volume, dotted=True))
        index += 1
    for fuel_i in range(HIDDEN_FUEL_COUNT):
        x, y = _next_open_tile(occupied, terrain, index)
        world["containers"].append(
            SimContainerDict(x=x, y=y, volume=_hidden_volume(fuel_i), dotted=False)
        )
        index += 1
    for _ in range(HIDDEN_EQUIPMENT_COUNT):
        x, y = _next_open_tile(occupied, terrain, index)
        world["equipment"].append(SimEquipmentDict(x=x, y=y))
        index += 1


def seed_practice_client(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    layout: PracticeLayout,
    client_id: int,
) -> None:
    """Place the client tank at the layout's real join spawn.

    The client boots rank 1 at a genuinely full tank (fuel capacity
    1100, all slots at the rank cap of 25) — under the 2026-07-25
    hunt-only-when-full contract that is the only state a session
    legally opens a fight from.

    Args:
        world: Simulated world (client tank added).
        terrain: Static terrain for the placement search.
        layout: The chosen practice layout.
        client_id: The connected client's tank id.

    Raises:
        RuntimeError: If no open tile exists near the layout spawn.
    """
    spawn_x, spawn_y = layout["client_spawn"]
    spot = find_open_tile_near(
        world, terrain, spawn_x, spawn_y, world["tick"], min_radius=0, max_radius=6
    )
    if spot is None:
        raise RuntimeError(
            f"seed_practice_client: no open tile near layout spawn ({spawn_x},{spawn_y})"
        )
    client = make_sim_tank(
        client_id, 2, _CLIENT_RANK, spot[0], spot[1], fuel_capacity(_CLIENT_RANK)
    )
    client["counts"] = [_CLIENT_COUNTS] * 5
    world["tanks"][client_id] = client


__all__ = [
    "DOTTED_FUEL_COUNT",
    "HIDDEN_DRAINED_PERIOD",
    "HIDDEN_EQUIPMENT_COUNT",
    "HIDDEN_FUEL_COUNT",
    "LARGE_VOLUME_CYCLE",
    "MINE_DENSITY",
    "MINE_SHAPE_CYCLE",
    "MINE_TEAM_CYCLE",
    "PRACTICE_LAYOUTS",
    "SMALL_VOLUME_CYCLE",
    "PracticeLayout",
    "seed_field_population",
    "seed_minefield",
    "seed_practice_client",
    "select_practice_layout",
]
