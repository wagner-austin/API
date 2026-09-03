"""Seed a sim room: containers, population, and the practice client.

The dotted and hidden container layouts, the field population, and the
practice-room client. Mines and ferries are
:mod:`tankpit_bot.sim.world_seed_mines`.
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
)


class PracticeLayout(TypedDict):
    """One real practice-room state lifted from an archive capture."""

    provenance: str
    client_spawn: tuple[int, int]
    roster: tuple[tuple[int, int, int, int, int], ...]


DOTTED_FUEL_COUNT = 620

DOTTED_STOCKED_PERIOD = 5

HIDDEN_FUEL_COUNT = 840

HIDDEN_DRAINED_PERIOD = 2

HIDDEN_EQUIPMENT_COUNT = 180

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

SMALL_VOLUME_CYCLE: tuple[int, ...] = (34, 57, 120, 180, 250, 320, 400, 460)

SMALL_PERIOD = 5

_CLIENT_RANK = 1

_CLIENT_COUNTS = 25

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

_MAP_SPAN = 256

_TILE_COUNT = _MAP_SPAN * _MAP_SPAN

_SEED_STRIDE = 97

_PROBE_STRIDE = 251


PRACTICE_LAYOUT_PROVENANCES: tuple[str, ...] = tuple(
    layout["provenance"] for layout in PRACTICE_LAYOUTS
)
"""Every layout's provenance, in table order — the names a caller may ask for.

Derived from the table rather than written beside it, so a layout added to
:data:`PRACTICE_LAYOUTS` is selectable by name without a second edit that
could be forgotten.
"""


class UnknownPracticeLayoutError(LookupError):
    """Raised when a named layout is not in :data:`PRACTICE_LAYOUTS`."""


def layout_by_provenance(provenance: str) -> PracticeLayout:
    """Pick the layout a caller NAMED, rather than one derived from a stamp.

    This is the selector a sweep member uses. :func:`select_practice_layout`
    varies the world with the run's name, which is right for interactive
    variety and wrong for a measurement — see its docstring.

    Args:
        provenance: The layout's ``provenance``, one of
            :data:`PRACTICE_LAYOUT_PROVENANCES`.

    Returns:
        The named layout.

    Raises:
        UnknownPracticeLayoutError: When no layout carries that provenance.
            Raised rather than falling back to a default, because a sweep
            that silently played a different world than the one its member
            document names would produce numbers nobody could interpret --
            which is the failure this selector exists to remove.
    """
    for layout in PRACTICE_LAYOUTS:
        if layout["provenance"] == provenance:
            return layout
    raise UnknownPracticeLayoutError(
        f"no practice layout named {provenance!r}; known: {list(PRACTICE_LAYOUT_PROVENANCES)}"
    )


def select_practice_layout(stamp: str) -> PracticeLayout:
    """Pick a layout deterministically from a run's stamp, for VARIETY.

    **Never reach this from a sweep member.** The stamp is a run's NAME, and
    this function makes it a world INPUT: naming a run changes what it
    plays. That is wanted interactively -- successive local soaks see
    different rooms for free -- and it is a confound in any measurement,
    because an array whose tasks stamp themselves varies the map together
    with whatever parameter the sweep meant to vary.

    It cost a published result on 2026-09-01: a saturation table was
    measured across session depths whose stamps moved with them, so the
    layout moved too, and the conclusion had to be retracted.

    A sweep member names its world with :func:`layout_by_provenance` and
    passes its own population seed, leaving the stamp to be nothing but a
    label.

    Args:
        stamp: The run stamp (any string).

    Returns:
        One of :data:`PRACTICE_LAYOUTS` — same stamp, same layout.
    """
    return PRACTICE_LAYOUTS[zlib.crc32(stamp.encode("utf-8")) % len(PRACTICE_LAYOUTS)]


def population_seed_for_stamp(stamp: str) -> int:
    """Derive the container-population seed a stamp implies, for VARIETY.

    The same hazard as :func:`select_practice_layout` and the less visible
    half of it: this seed decides where every container lies, and nothing
    logs it, so a stamp-varied larder moves a forage or economy measurement
    with no line in any artifact saying it did.

    Named here rather than left inline at the call site so the derivation is
    something a sweep can be seen NOT to use.

    Args:
        stamp: The run stamp (any string).

    Returns:
        The seed — same stamp, same population.
    """
    return zlib.crc32(stamp.encode("utf-8"))


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
    "DOTTED_STOCKED_PERIOD",
    "HIDDEN_DRAINED_PERIOD",
    "HIDDEN_EQUIPMENT_COUNT",
    "HIDDEN_FUEL_COUNT",
    "LARGE_VOLUME_CYCLE",
    "PRACTICE_LAYOUTS",
    "PRACTICE_LAYOUT_PROVENANCES",
    "SMALL_PERIOD",
    "SMALL_VOLUME_CYCLE",
    "PracticeLayout",
    "UnknownPracticeLayoutError",
    "layout_by_provenance",
    "population_seed_for_stamp",
    "seed_field_population",
    "seed_practice_client",
    "select_practice_layout",
]
