"""The gun probe, driven end to end against a scripted game."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.gun_probe import EXIT_BAD_USAGE, EXIT_NO_TURRET, EXIT_OK, main

from tests.wire_fixtures import ScriptedPeer, StubbedConnect, entity, lines, option, sample

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"
_PLACEMENT_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"


def _args(samples: str = "1") -> list[str]:
    return ["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), samples]


@pytest.mark.parametrize("argv", [[], ["27200"], ["a", "b", "c", "d", "e"]])
def test_a_bad_argument_count_prints_usage(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: gun_probe")


def test_the_module_entry_point_exits_with_the_probe_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.gun_probe")
    sys.argv = ["gun_probe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.gun_probe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.gun_probe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: gun_probe")


def test_a_standing_turrets_offers_and_the_ladder_verdict_are_reported(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The probe's whole purpose: the option rows and the channel's own
    decision on the identical sample, side by side."""
    world = sample(
        entity(213, "commandCenter"),
        entity(42, "c_turret_t1", x=100.0),
        credits=4000,
        options=(
            option(42, "c_turret_t2_gun", key="u_t2", price=1000),
            # Another unit's option: skipped by the report, exactly as the
            # centre's own actions are in a live sample.
            option(213, "builder", key="u_builder", index=1, price=500),
        ),
    )
    peer = ScriptedPeer(lines(world, world, world))
    with StubbedConnect(peer):
        assert main(_args()) == EXIT_OK
    out = capsys.readouterr().out
    assert "turrets standing: 1" in out
    assert "turret 42: complete=True queued=0" in out
    assert "option c_turret_t2_gun" in out
    assert "'type_name': 'c_turret_t2_gun'" in out


def test_a_world_without_turrets_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    world = sample(entity(213, "commandCenter"), credits=4000)
    peer = ScriptedPeer(lines(world, world, world))
    with StubbedConnect(peer):
        assert main(_args()) == EXIT_NO_TURRET
    printed = capsys.readouterr().out.splitlines()
    assert printed[-2:] == ["turrets standing: 0", "ladder verdict: []"]
