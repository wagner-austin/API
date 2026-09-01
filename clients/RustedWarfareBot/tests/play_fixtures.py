"""Shared fixtures for the play entry point's tests.

The catalogue and placement paths are the real archived dumps, so prices and
placement rules asserted anywhere are the engine's own. Split from
test_play.py when the opening-compile tests pushed it past the size cap
(log 2026-08-06); the fixtures are the shared half.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"
PLACEMENT_PATH = PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"

#: What the Builder offers by default here -- the plan's own types, which the
#: live capture confirms unit 214 reports.
BUILDER_OFFERS = ("extractorT1", "landFactory", "c_tank")

BUILDER = (214, "builder")

#: A boot-sandbox roster: rich enough to trip the swap wait, factory included
#: so a plan compiled against it would insert nothing.
SANDBOX = (
    (100, "commandCenter"),
    (101, "builder"),
    (102, "landFactory"),
    (103, "gunShip"),
    (104, "airFactory"),
)

#: What DEFAULT_GOALS expands to against the real tree from a fresh opening
#: roster. Stated here so a change to either the goals or the tree fails
#: loudly rather than quietly changing what these tests drive.
EXPANDED = (
    "extractorT1",
    "extractorT1",
    "extractorT1",
    "landFactory",
    "c_tank",
    "c_tank",
    "c_tank",
    "c_tank",
)


def entity_line(frame: int, index: int, unit_id: int, type_name: str) -> str:
    """Render one entity record as the wire would carry it."""
    return (
        f'{{"kind":"entity","frame":{frame},"index":{index},"id":{unit_id},'
        f'"type":"{type_name}","class":"units.x","x":100.0,"y":200.0,'
        f'"team":0,"mine":true,"hostile":false,"movement":"LAND","group":1,'
        f'"flying":false,"submerged":false,"touching_water":false,'
        f'"hp":100.0,"max_hp":100.0,"complete":true,"queued":0,"damaged_by":""}}'
    )


def pool_line(frame: int, index: int, tile_x: int, tile_y: int) -> str:
    """Render one resource pool record as the wire would carry it."""
    return (
        f'{{"kind":"pool","frame":{frame},"index":{index},'
        f'"tile_x":{tile_x},"tile_y":{tile_y},'
        f'"x":{tile_x * 20 + 10}.0,"y":{tile_y * 20 + 10}.0,"group_land":1}}'
    )


def option_line(frame: int, index: int, unit_id: int, produces: str) -> str:
    """Render one build option record as the wire would carry it."""
    return (
        f'{{"kind":"option","frame":{frame},"index":{index},"unit_id":{unit_id},'
        f'"produces":"{produces}","key":"u_x","placed":true,"available":true,'
        f'"makes_something":true,"price":100}}'
    )


def sample_lines(
    frame: int,
    credits: int,
    *entities: tuple[int, str],
    pools: tuple[tuple[int, int], ...] = (),
    options: tuple[tuple[int, str], ...] | None = None,
) -> list[str]:
    """Render one whole sample as wire lines."""
    if options is None:
        options = tuple((214, name) for name in BUILDER_OFFERS)
    lines = [
        f'{{"kind":"frame","frame":{frame},"clock_ms":{frame * 3},'
        f'"visible":{len(entities)},"pools":{len(pools)},'
        f'"options":{len(options)},"players":0,"refused":0,'
        f'"credits":{credits},"defeated":false,"wiped":false,"players_left":6}}'
    ]
    for index, (unit_id, type_name) in enumerate(entities):
        lines.append(entity_line(frame, index, unit_id, type_name))
    for index, (tile_x, tile_y) in enumerate(pools):
        lines.append(pool_line(frame, index, tile_x, tile_y))
    for index, (unit_id, produces) in enumerate(options):
        lines.append(option_line(frame, index, unit_id, produces))
    return lines


__all__ = [
    "BUILDER",
    "BUILDER_OFFERS",
    "CATALOGUE_PATH",
    "EXPANDED",
    "PLACEMENT_PATH",
    "SANDBOX",
    "entity_line",
    "option_line",
    "pool_line",
    "sample_lines",
]
