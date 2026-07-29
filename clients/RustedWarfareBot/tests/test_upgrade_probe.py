"""The upgrade probe, driven end to end against a scripted game."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.upgrade_probe import EXIT_BAD_USAGE, EXIT_NO_STRUCTURE, EXIT_OK, main

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
    assert capsys.readouterr().out.startswith("usage: upgrade_probe")


def test_the_module_entry_point_exits_with_the_probe_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.upgrade_probe")
    sys.argv = ["upgrade_probe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.upgrade_probe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.upgrade_probe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: upgrade_probe")


def test_a_standing_structures_offer_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    """The probe's whole purpose: ask the engine, not the build tree."""
    world = sample(
        entity(213, "commandCenter"),
        entity(400, "extractorT1"),
        credits=4000,
        options=(option(400, "extractorT2", placed=True, available=False),),
    )
    peer = ScriptedPeer(lines(world, world, world))
    with StubbedConnect(peer):
        assert main(_args()) == EXIT_OK
    out = capsys.readouterr().out
    assert "extractors standing: 1" in out
    assert "extractorT2" in out


def test_a_world_whose_structures_offer_nothing_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The measured outcome in a real match, and it is a result rather than a
    failure to observe ([[policy-holding-ground]]).
    """
    # Structures are standing -- they simply carry no options, which is exactly
    # what four extractors did in a live match.
    world = sample(entity(213, "commandCenter"), entity(400, "extractorT1"), credits=4000)
    peer = ScriptedPeer(lines(world, world, world))
    with StubbedConnect(peer):
        assert main(_args()) == EXIT_NO_STRUCTURE
    printed = capsys.readouterr().out.splitlines()
    assert printed[-2:] == [
        "extractors standing: 1",
        "no owned structure offered any option at all",
    ]
