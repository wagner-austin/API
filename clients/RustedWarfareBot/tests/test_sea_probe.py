"""The sea probe, driven end to end against scripted water.

Terrain discovery by attempt, on a fake shore: the engine's accept or
ignore is the sensor, so the scripted game refuses the first candidate by
silence and accepts the second by growing a factory.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.sea_probe import EXIT_BAD_USAGE, EXIT_NO_WATER, EXIT_OK, PATIENCE, main

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
    assert capsys.readouterr().out.startswith("usage: sea_probe")


def test_the_module_entry_point_exits_with_the_probe_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.sea_probe")
    sys.argv = ["sea_probe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.sea_probe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.sea_probe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: sea_probe")


def test_a_world_without_a_builder_reports_and_declines() -> None:
    world = sample(_ANCHOR, pools=(_POOL,))
    peer = ScriptedPeer(lines(world))
    with StubbedConnect(peer):
        assert main(_args("1")) == EXIT_NO_WATER


def test_a_refused_candidate_advances_and_an_accepted_one_stands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The probe's whole mechanism: the first fraction is offered PATIENCE
    times and never grows a factory (dry land, by scripted silence); the
    next acceptance appears as a factory entity, completes, publishes its
    options, and the ordered submarine enters the world."""
    dry = sample(_ANCHOR, _BUILDER, pools=(_POOL,))
    standing = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, "seaFactory", x=250.0, y=0.0),
        pools=(_POOL,),
        options=(option(9, "attackSubmarine", key="p_sub", price=800),),
    )
    afloat = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, "seaFactory", x=250.0, y=0.0),
        entity(11, "attackSubmarine", x=260.0, y=0.0),
        pools=(_POOL,),
    )
    worlds = [dry] * PATIENCE + [standing, standing, afloat]
    peer = ScriptedPeer(lines(*worlds))
    with StubbedConnect(peer):
        assert main(_args(str(len(worlds)))) == EXIT_OK
    out = capsys.readouterr().out
    assert "fraction 0.20: refused" in out
    assert "STANDS at fraction 0.25" in out
    assert "options: attackSubmarine[avail=True price=800]" in out
    assert "SUBMARINE afloat: unit 11" in out
    builds = [line for line in peer.sent if '"kind":"build"' in line and "seaFactory" in line]
    # Exactly PATIENCE offers at the first fraction; the scripted accept
    # arrives before the second fraction needs any.
    assert len(builds) == PATIENCE
    assert builds[0] == '{"kind":"build","unit_id":2,"x":200.0,"y":0.0,"type":"seaFactory"}'


def test_every_fraction_refused_reports_no_water(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dry = sample(_ANCHOR, _BUILDER, pools=(_POOL,))
    count = PATIENCE * 9 + 2
    peer = ScriptedPeer(lines(*([dry] * count)))
    with StubbedConnect(peer):
        # One sample is read before the loop, so the budget leaves it room.
        assert main(_args(str(count - 1))) == EXIT_NO_WATER
    out = capsys.readouterr().out
    refusals = [line for line in out.splitlines() if "refused after" in line]
    assert len(refusals) == 9
    assert out.splitlines()[-1] == "[sea] done: placed_at=-1.00 produced=False"
