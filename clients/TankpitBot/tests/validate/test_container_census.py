"""The container census: scan footprints, per-field records, and the held-out score."""

from __future__ import annotations

import math
from collections.abc import Generator
from pathlib import Path

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.resources import data_directory
from tankpit_bot.validate.container_census import (
    CensusError,
    FieldTally,
    block_dispersion,
    census_field,
    run_census,
    tally_capture,
    transfer_score,
)
from tankpit_bot.validate.container_census_types import (
    FieldCensusDict,
    TerrainCountsDict,
    TransferScoreDict,
)
from tests.analysis._capture_fixtures import _session_json
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.validate._census_captures import (
    as_replay,
    capture_text,
    introduced,
    joined,
    map_open,
    radar,
    ranked,
    received,
    scanned,
    spent_radar,
)


@pytest.fixture()
def terrain_loads() -> Generator[list[Path], None, None]:
    """Load every field as open ground, recording which GIF was asked for.

    Yields:
        The list every terrain load appends its path to.
    """
    original = _test_hooks.load_terrain_map
    loads: list[Path] = []

    def load_open_terrain(gif_path: Path) -> TerrainMapProtocol:
        """An all-ground field."""
        loads.append(gif_path)
        return InMemoryTerrainMap()

    _test_hooks.load_terrain_map = load_open_terrain
    yield loads
    _test_hooks.load_terrain_map = original


def test_scans_reveal_the_sim_footprint_and_count_what_they_list() -> None:
    """A free radar reveals the rank square, an extra one the whole window.

    The client stands at (10, 10) at rank 0 (radius 2) with its window at
    (2, 2). Only lone-radar ticks whose batch carries a 0x4F are read.
    """
    tally = FieldTally()
    tally_capture(
        as_replay(
            [
                introduced(1000, 10, 10, rank=0, left=2, top=2),
                radar(1100),
                received(3000, scanned((11, 10, 500), (12, 12, -1), (40, 40, 100))),
                radar(3100),
                received(5000, spent_radar(), scanned((3, 3, 0))),
                map_open(5100),
                received(7000, scanned((4, 4, 500))),
                radar(7100),
                map_open(7101),
                received(9000, scanned((5, 5, 500))),
                radar(9100),
                received(11000, ranked(0)),
            ]
        ),
        tally,
    )
    assert (tally.captures, tally.scans, tally.tile_reads) == (1, 2, 25 + 256)
    assert (tally.container_reads, tally.footprint_misses) == (3, 1)
    assert tally.observed == {(x, y) for x in range(2, 18) for y in range(2, 18)}
    assert (tally.fuel, tally.equipment) == ({(11, 10), (3, 3)}, {(12, 12)})
    assert tally.sites() == {(11, 10), (3, 3), (12, 12)}


def test_a_window_at_the_map_edge_reveals_only_tiles_on_the_map() -> None:
    """An extra radar from a window hanging past column 255 reveals 8 x 8."""
    tally = FieldTally()
    tally_capture(
        as_replay(
            [
                introduced(1000, 250, 250, rank=0, left=248, top=248),
                radar(1100),
                received(3000, spent_radar(), scanned()),
            ]
        ),
        tally,
    )
    assert (tally.scans, tally.tile_reads, len(tally.observed)) == (1, 64, 64)


def test_a_scan_before_the_client_is_placed_is_not_counted() -> None:
    """Without a window, tile and rank there is no footprint to read."""
    tally = FieldTally()
    tally_capture(as_replay([radar(100), received(3000, scanned((1, 1, 500)))]), tally)
    assert (tally.captures, tally.scans, tally.observed) == (0, 0, set())


def test_block_dispersion_compares_blocks_with_a_uniform_rate() -> None:
    """Two open blocks holding 2 and 0 of 2 sites read chi-square 2 on one df."""
    observed = {(x, y) for x in range(32) for y in range(16)}
    rock = {(x, y) for x in range(32, 48) for y in range(16)}
    terrain = InMemoryTerrainMap(terrain_data=dict.fromkeys(rock, "#"))
    assert block_dispersion(observed | rock, {(1, 1), (2, 2)}, terrain) == 2.0
    assert block_dispersion(observed, set(), terrain) == 0.0
    one_block = {(x, y) for x in range(16) for y in range(16)}
    assert block_dispersion(one_block, {(1, 1)}, terrain) == 0.0


def test_a_field_record_splits_observations_and_sites_by_terrain() -> None:
    """Ground, water and rock are counted apart; kinds are counted apart."""
    tally = FieldTally()
    tally.captures, tally.scans, tally.footprint_misses = 2, 3, 1
    tally.tile_reads, tally.container_reads = 40, 4
    tally.observed = {(0, 0), (1, 0), (2, 0)}
    tally.fuel = {(0, 0)}
    tally.equipment = {(1, 0)}
    terrain = InMemoryTerrainMap(terrain_data={(1, 0): "W", (2, 0): "#"})
    assert census_field("field07.gif", tally, terrain) == FieldCensusDict(
        field="field07.gif",
        captures=2,
        scans=3,
        footprint_misses=1,
        tile_reads=40,
        container_reads=4,
        observed=TerrainCountsDict(ground=1, water=1, rock=1),
        sites=TerrainCountsDict(ground=1, water=1, rock=0),
        equipment_sites=1,
        fuel_sites=1,
        block_dispersion=0.0,
    )


def _field(
    name: str,
    observed: tuple[int, int, int],
    sites: tuple[int, int, int],
    reads: tuple[int, int] = (10, 1000),
) -> FieldCensusDict:
    """A field record from (ground, water, rock) counts and (container, tile) reads."""
    return FieldCensusDict(
        field=name,
        captures=1,
        scans=1,
        footprint_misses=0,
        tile_reads=reads[1],
        container_reads=reads[0],
        observed=TerrainCountsDict(ground=observed[0], water=observed[1], rock=observed[2]),
        sites=TerrainCountsDict(ground=sites[0], water=sites[1], rock=sites[2]),
        equipment_sites=0,
        fuel_sites=0,
        block_dispersion=0.0,
    )


def test_the_shape_is_fitted_on_one_field_and_scored_on_another() -> None:
    """Water at 0.8 of ground and no rock, with only the level refitted."""
    train = _field("field01.gif", (100, 50, 20), (20, 8, 0))
    test = _field("field05.gif", (200, 100, 50), (30, 12, 0), reads=(7, 700))
    constant = -(42 * math.log(0.12) + 308 * math.log(0.88)) / 350
    shape = (
        -(30 * math.log(0.15) + 170 * math.log(0.85) + 12 * math.log(0.12) + 88 * math.log(0.88))
        / 350
    )
    assert transfer_score(train, test) == pytest.approx(
        TransferScoreDict(
            train_field="field01.gif",
            test_field="field05.gif",
            water_factor=0.8,
            train_density=0.01,
            test_density=0.01,
            log_loss_constant=constant,
            log_loss_shape=shape,
        )
    )


def test_a_train_field_without_water_or_rock_fits_both_at_zero() -> None:
    """Unobserved classes read as holding nothing."""
    score = transfer_score(_field("a", (10, 0, 0), (2, 0, 0)), _field("b", (10, 0, 0), (1, 0, 0)))
    assert score["water_factor"] == 0.0
    assert score["log_loss_shape"] == pytest.approx(score["log_loss_constant"])


def test_shapes_that_cannot_score_the_test_field_are_refused() -> None:
    """No ground site, a site the shape forbids, and a certain absence that is not."""
    with pytest.raises(CensusError, match="CENSUS_NO_GROUND: a"):
        transfer_score(_field("a", (10, 5, 5), (0, 1, 0)), _field("b", (10, 0, 0), (1, 0, 0)))
    with pytest.raises(CensusError, match="CENSUS_IMPOSSIBLE_SITE: 1/5"):
        transfer_score(_field("a", (10, 5, 5), (2, 0, 0)), _field("b", (10, 0, 5), (1, 0, 1)))
    with pytest.raises(CensusError, match="CENSUS_IMPOSSIBLE_SITE: 0/1"):
        transfer_score(_field("a", (10, 10, 0), (2, 4, 0)), _field("b", (10, 1, 0), (10, 0, 0)))


def _scanning(image: str, *containers: tuple[int, int, int]) -> str:
    """A capture on a field whose one scan lists these containers."""
    return capture_text(
        [
            *joined(image),
            introduced(1000, 10, 10, rank=0, left=2, top=2),
            radar(1100),
            received(3000, scanned(*containers)),
        ]
    )


def test_a_run_censuses_each_known_field_and_scores_every_usable_pair(
    tmp_path: Path, terrain_loads: list[Path]
) -> None:
    """Undecodable, fieldless, unknown-field and scanless captures add nothing."""
    files = {
        "a-no-magic": _session_json(magic=None),
        "b-no-room": capture_text([introduced(1000, 10, 10, rank=0, left=2, top=2)]),
        "c-unknown-field": _scanning("field99.gif", (11, 10, 500)),
        "d-scanless": capture_text(joined("field02.gif")),
        "e-field01": _scanning("field01.gif", (11, 10, 500)),
        "f-field01": _scanning("field01.gif", (12, 10, -1)),
        "g-field05": _scanning("field05.gif"),
    }
    paths = []
    for name, text in files.items():
        path = tmp_path / f"{name}.capture_session.json"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    report = run_census(paths)
    assert [(f["field"], f["captures"], f["scans"]) for f in report["fields"]] == [
        ("field01.gif", 2, 2),
        ("field05.gif", 1, 1),
    ]
    assert report["fields"][0]["sites"] == TerrainCountsDict(ground=2, water=0, rock=0)
    assert terrain_loads == [data_directory() / "field01_r.gif", data_directory() / "field05_r.gif"]
    assert [(t["train_field"], t["test_field"]) for t in report["transfers"]] == [
        ("field01.gif", "field05.gif")
    ]
    assert report["transfers"][0]["log_loss_constant"] == 0.0
