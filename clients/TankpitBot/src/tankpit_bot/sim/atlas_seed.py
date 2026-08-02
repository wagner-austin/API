"""Seed the sim's container field from the mined longitudinal atlas.

The archive-mined atlas (``analysis_scripts/mine_container_atlas.py``,
[[game-economy]] 2026-08-01) is the REAL field: every per-tile
container statement from every recorded session, cross-session
persistence-verified at 94.9%+ over a month. Seeding from it replaces
the statistical density model with the actual room — roughly 5,000+
stocked tiles where the model seeded ~670 — so soak economics (larder
yields, forage returns, ``no_productive_collect`` horizons) match live
play.

Classification per atlas tile (the mined vocabulary):

* ``last_v == -1`` — equipment container.
* ``last_v > 0`` — fuel container at the last-read volume; ``dotted``
  when the visible layer ever showed it (a 0x5A/0x43 sighting means
  the container is EXPOSED on the map).
* ``last_v == 0`` and visible-seen — a drained DOT: part of the real
  map experience (dots that answer pickups with 0x52 code 4), seeded
  only when the drain was read within :data:`DRAINED_DOT_FRESH_MS`
  of the atlas's newest observation (stale exposure history bloats
  the map far past the live ~620-1,077 dot census).
* ``last_v == 0`` radar-only — an empty hidden tile: nothing to seed.

Tiles on rock are mining artifacts and are skipped with a tally;
tiles on WATER are kept — the water-locked population is real
(12 of 13 believed containers in run bot-20260728-093011) and is
exactly what the ferry doctrine harvests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
)

from tankpit_bot import _test_hooks
from tankpit_bot.sim.world import SimContainerDict, SimEquipmentDict, SimWorldDict

DEFAULT_ATLAS_PATH = Path("runs") / "analysis" / "container_atlas.json"

DRAINED_DOT_FRESH_MS = 7 * 24 * 3_600_000
"""How recent a drained dot's last read must be to seed it.

The atlas accumulates 120 days of exposure history (6,675 drained
tiles), but the LIVE dot census runs ~620-1,077 — exposure memory
fades as the field turns over. Seeding the full history quadrupled
the believed-container registry and dragged long sessions superlinear
(120 ghost+atlas rounds took 172 s). A week matches the measured
~95% persistence window ([[game-economy]])."""


class AtlasSeedTallyDict(TypedDict):
    """What one atlas seeding placed and skipped."""

    fuel: int
    drained_dots: int
    equipment: int
    water_tiles: int
    rock_skipped: int


def _field_key(atlas: JSONObject, field: str) -> str | None:
    """Find the atlas room entry whose field image matches the world's.

    The world names the terrain GIF (``field01_r.gif``); the atlas
    keys are ``room|field01.gif`` from the lobby ROOM_LIST rows.

    Args:
        atlas: Decoded atlas document.
        field: The sim world's field GIF name.

    Returns:
        The matching atlas key, or ``None``.
    """
    base = Path(field).name.replace("_r.gif", ".gif")
    for key in sorted(atlas):
        if key.split("|", 1)[-1] == base:
            return key
    return None


def _exposure(
    entry: JSONObject,
    x: int,
    y: int,
    dotted_tiles: frozenset[tuple[int, int]] | None,
    newest_ms: int,
) -> tuple[bool, bool]:
    """Resolve one tile's exposure: (dotted, seed-as-drained-dot).

    Args:
        entry: The atlas tile entry.
        x: Tile X.
        y: Tile Y.
        dotted_tiles: The exact exposed set (ghost mode), or ``None``
            for the visible-seen + drained-freshness heuristics.
        newest_ms: The atlas's newest observation timestamp.

    Returns:
        Whether a stocked seed is dotted, and whether a drained tile
        seeds as a dot at all.
    """
    if dotted_tiles is not None:
        dotted = (x, y) in dotted_tiles
        return dotted, dotted
    visible = entry["visible_seen"] is True
    drained_dot = (
        visible and newest_ms - narrow_json_to_int(entry["last_ms"]) <= DRAINED_DOT_FRESH_MS
    )
    return visible, drained_dot


def seed_atlas_population(
    world: SimWorldDict,
    terrain: _test_hooks.TerrainMapProtocol,
    atlas_path: Path,
    dotted_tiles: frozenset[tuple[int, int]] | None = None,
) -> AtlasSeedTallyDict:
    """Populate the world's containers from the mined atlas.

    Args:
        world: Simulated world (mutated: containers and equipment
            appended).
        terrain: Static terrain of the world's field (rock filter and
            the water tally).
        atlas_path: The mined ``container_atlas.json``.
        dotted_tiles: When given (ghost mode), the EXACT exposed set —
            the recording's own 0x4C dot atlas: only these tiles seed
            dotted (including their drained dots), everything else
            seeds hidden. Without it the miner's visible-seen +
            drained-freshness heuristics apply.

    Returns:
        The seeding tally.

    Raises:
        RuntimeError: If the atlas is unreadable or holds no entry for
            the world's field — an atlas run needs the real data,
            loudly.
    """
    try:
        atlas = narrow_json_to_dict(load_json_str(_test_hooks.read_text(atlas_path)))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"atlas {atlas_path} unreadable: {error}") from error
    key = _field_key(atlas, world["field"])
    if key is None:
        raise RuntimeError(
            f"atlas {atlas_path} has no entry for field {world['field']!r} (keys: {sorted(atlas)})"
        )
    occupied = {(tank["x"], tank["y"]) for tank in world["tanks"].values() if tank["alive"]}
    tally = AtlasSeedTallyDict(fuel=0, drained_dots=0, equipment=0, water_tiles=0, rock_skipped=0)
    entries = narrow_json_to_dict(atlas[key])
    newest_ms = max(
        (narrow_json_to_int(narrow_json_to_dict(raw)["last_ms"]) for raw in entries.values()),
        default=0,
    )
    for tile_key, raw_entry in sorted(entries.items()):
        entry = narrow_json_to_dict(raw_entry)
        x_text, y_text = tile_key.split(",")
        x, y = int(x_text), int(y_text)
        if (x, y) in occupied:
            continue
        on_ground = terrain.is_passable(x, y)
        on_water = not on_ground and terrain.get_terrain(x, y) == terrain.WATER
        if not on_ground and not on_water:
            tally["rock_skipped"] += 1
            continue
        if on_water:
            tally["water_tiles"] += 1
        last_v = narrow_json_to_int(entry["last_v"])
        dotted, drained_dot = _exposure(entry, x, y, dotted_tiles, newest_ms)
        if last_v == -1:
            world["equipment"].append(SimEquipmentDict(x=x, y=y))
            tally["equipment"] += 1
        elif last_v > 0:
            world["containers"].append(SimContainerDict(x=x, y=y, volume=last_v, dotted=dotted))
            tally["fuel"] += 1
        elif drained_dot:
            world["containers"].append(SimContainerDict(x=x, y=y, volume=0, dotted=True))
            tally["drained_dots"] += 1
    return tally


__all__ = [
    "DEFAULT_ATLAS_PATH",
    "DRAINED_DOT_FRESH_MS",
    "AtlasSeedTallyDict",
    "seed_atlas_population",
]
