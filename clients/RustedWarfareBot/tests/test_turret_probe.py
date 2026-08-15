"""The turret probe, driven end to end against scripted shore and blood.

Terrain discovery by attempt, on a fake shoreline: the engine's accept or
ignore is the sensor, the conversion is priced by the option row, and the
blood proof is the ``damaged_by`` attribution the live run carried
(log 2026-08-13).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.turret_probe import (
    BATTERY,
    EXIT_BAD_USAGE,
    EXIT_NO_GROUND,
    EXIT_OK,
    PATIENCE,
    TURRET,
    main,
)

from tests.wire_fixtures import ScriptedPeer, StubbedConnect, entity, lines, option, pool, sample

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"
_PLACEMENT_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"

_ANCHOR = entity(1, "commandCenter", x=0.0, y=0.0)
_BUILDER = entity(2, "builder", x=10.0, y=0.0)
_POOL = pool(x=500.0, y=0.0)


def _args(samples: str) -> list[str]:
    return ["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), samples]


@pytest.mark.parametrize("argv", [[], ["27200"], ["a", "b", "c", "d", "e"]])
def test_a_bad_argument_count_prints_usage(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: turret_probe")


def test_the_module_entry_point_exits_with_the_probe_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.turret_probe")
    sys.argv = ["turret_probe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.turret_probe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.turret_probe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: turret_probe")


def test_a_world_without_a_builder_reports_and_declines() -> None:
    world = sample(_ANCHOR, pools=(_POOL,))
    peer = ScriptedPeer(lines(world))
    with StubbedConnect(peer):
        assert main(_args("1")) == EXIT_NO_GROUND


def test_the_whole_mechanism_stands_converts_and_draws_blood(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The probe's full arc on scripted terrain: the water-most fraction
    is refused by silence, the next accepts a turret, the fork is ordered
    off the option row's price, the battery stands, a hostile enters
    reach, and the engine's own attribution names the battery."""
    dry = sample(_ANCHOR, _BUILDER, pools=(_POOL,))
    growing = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, TURRET, x=205.0, y=0.0, complete=False),
        pools=(_POOL,),
    )
    standing = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, TURRET, x=205.0, y=0.0),
        pools=(_POOL,),
        options=(option(9, BATTERY, key="u_arty", price=1600),),
    )
    forked = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, BATTERY, x=205.0, y=0.0),
        entity(21, "gunBoat", x=400.0, y=0.0, mine=False, hostile=True, movement="WATER"),
        pools=(_POOL,),
    )
    bloodied = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, BATTERY, x=205.0, y=0.0),
        entity(
            21,
            "gunBoat",
            x=400.0,
            y=0.0,
            mine=False,
            hostile=True,
            movement="WATER",
            hp=90.0,
            max_hp=170.0,
            damaged_by=BATTERY,
        ),
        pools=(_POOL,),
    )
    worlds = [dry] * PATIENCE + [growing, standing, standing, forked, bloodied, bloodied]
    peer = ScriptedPeer(lines(*worlds))
    with StubbedConnect(peer):
        # One sample is read before the loop, so the budget leaves it room.
        assert main(_args(str(len(worlds) - 1))) == EXIT_OK
    out = capsys.readouterr().out
    assert "fraction 0.22: refused" in out
    assert "turret STANDS at fraction 0.20" in out
    assert f"turret offers: {BATTERY}[avail=True price=1600]" in out
    assert "BATTERY STANDS: unit 9" in out
    assert "in reach: gunBoat (WATER) at 195" in out
    assert "DREW BLOOD: gunBoat (WATER)" in out
    assert "bled=gunBoat" in out
    produces = [line for line in peer.sent if '"kind":"produce"' in line and BATTERY in line]
    assert produces == [f'{{"kind":"produce","unit_id":9,"type":"{BATTERY}"}}'] * 2


def test_every_fraction_refused_reports_no_ground(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dry = sample(_ANCHOR, _BUILDER, pools=(_POOL,))
    count = PATIENCE * 7 + 2
    peer = ScriptedPeer(lines(*([dry] * count)))
    with StubbedConnect(peer):
        # One sample is read before the loop, so the budget leaves it room.
        assert main(_args(str(count - 1))) == EXIT_NO_GROUND
    out = capsys.readouterr().out
    refusals = [line for line in out.splitlines() if "refused after" in line]
    assert len(refusals) == 7
    done = "[battery] done: placed_at=-1.00 battery=False fate=gone bled=(nothing)"
    assert out.splitlines()[-1] == done
