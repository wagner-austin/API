"""``tankpit-container-census``: arguments, the printed census and the written JSON."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.validate.conformance_cli import EXIT_USAGE
from tankpit_bot.validate.container_census_cli import DEFAULT_CENSUS_OUT, format_census, main
from tankpit_bot.validate.container_census_types import (
    CensusReportDict,
    FieldCensusDict,
    TerrainCountsDict,
    TransferScoreDict,
    decode_census,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap
from tests.validate._census_captures import (
    capture_text,
    introduced,
    joined,
    radar,
    received,
    scanned,
)


@pytest.fixture()
def open_terrain() -> Generator[None, None, None]:
    """Load every field as open ground for the duration of a test."""
    original = _test_hooks.load_terrain_map

    def load_open_terrain(gif_path: Path) -> TerrainMapProtocol:
        """An all-ground field."""
        del gif_path
        return InMemoryTerrainMap()

    _test_hooks.load_terrain_map = load_open_terrain
    yield
    _test_hooks.load_terrain_map = original


def _archive(root: Path) -> None:
    """One field01 capture whose scan finds one fuel site, under ``root/bot``."""
    (root / "bot").mkdir(parents=True)
    (root / "bot" / "a.capture_session.json").write_text(
        capture_text(
            [
                *joined("field01.gif"),
                introduced(1000, 10, 10, rank=0, left=2, top=2),
                radar(1100),
                received(3000, scanned((11, 10, 500))),
            ]
        ),
        encoding="utf-8",
    )


def _written(out: Path) -> CensusReportDict:
    """The census a run wrote."""
    return decode_census(narrow_json_to_dict(load_json_str(out.read_text(encoding="utf-8"))))


def _printed(stdout: str) -> str:
    """Stdout from the census table's header on (seeding logs share stdout)."""
    return stdout[stdout.index("field ") :]


def test_the_printed_census_reads_rates_density_and_transfers() -> None:
    """An unobserved class reads as a dash; transfers print both losses."""
    field = FieldCensusDict(
        field="field01.gif",
        captures=3,
        scans=40,
        footprint_misses=1,
        tile_reads=1000,
        container_reads=7,
        observed=TerrainCountsDict(ground=400, water=0, rock=100),
        sites=TerrainCountsDict(ground=100, water=0, rock=0),
        equipment_sites=60,
        fuel_sites=40,
        block_dispersion=1.5,
    )
    transfer = TransferScoreDict(
        train_field="field01.gif",
        test_field="field05.gif",
        water_factor=0.794,
        train_density=0.0066,
        test_density=0.008,
        log_loss_constant=0.25075,
        log_loss_shape=0.23958,
    )
    lines = format_census(CensusReportDict(fields=[field], transfers=[transfer])).splitlines()
    assert lines[1:] == [
        f"{'field01.gif':<14} {3:>5} {40:>6} {500:>9} {100:>6} {'25.00%':>7} {'-':>7}"
        f" {0:>5} {'0.70%':>8} {'1.500':>6} {1:>5}",
        "",
        "terrain shape fitted on one field, scored on another (log loss per tile):",
        "  field01.gif -> field05.gif: water factor 0.794, shape 0.23958 vs constant 0.25075;"
        " density 0.66% -> 0.80%",
    ]


def test_bad_command_lines_are_refused_with_a_code(capsys: pytest.CaptureFixture[str]) -> None:
    """Unknown flags and flags without values exit with the usage code."""
    assert main(["--top", "3"]) == EXIT_USAGE
    assert capsys.readouterr().err == "CENSUS_USAGE: unknown argument '--top'\n"
    assert main(["--out"]) == EXIT_USAGE
    assert capsys.readouterr().err == "CENSUS_USAGE: --out needs a value\n"


def test_an_archive_with_no_scanned_field_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The empty census is still written, and the run says why it failed."""
    out = tmp_path / "census.json"
    assert main(["--root", str(tmp_path / "empty"), "--out", str(out)]) == 1
    empty = CensusReportDict(fields=[], transfers=[])
    assert capsys.readouterr().out == (
        format_census(empty)
        + f"\n\nwrote {out}\n"
        + "CENSUS_EMPTY: no capture under the roots scanned a known field\n"
    )
    assert _written(out) == empty


@pytest.mark.usefixtures("open_terrain")
def test_main_reads_sys_argv_and_defaults_to_the_spec_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no arguments the run reads runs/bot and runs/sniff and writes the default path."""
    _archive(tmp_path / "runs")
    original_argv = sys.argv
    original_cwd = Path.cwd()
    sys.argv = ["tankpit-container-census"]
    os.chdir(tmp_path)
    rc = main(None)
    os.chdir(original_cwd)
    sys.argv = original_argv
    assert rc == 0
    report = _written(tmp_path / DEFAULT_CENSUS_OUT)
    assert [(f["field"], f["sites"]["ground"]) for f in report["fields"]] == [("field01.gif", 1)]
    assert _printed(capsys.readouterr().out) == (
        format_census(report) + f"\n\nwrote {DEFAULT_CENSUS_OUT}\n"
    )
