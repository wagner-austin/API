"""The timeline probe: the teeing channel that records what the campaign saw.

Built to settle the zone screens' silent gun ladder, kept because the
question it answers -- what did the option stream actually carry, per
sample, during a real match -- recurs every time a channel goes quiet
(log 2026-08-04).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.gun_probe2 import EXIT_BAD_USAGE, EXIT_OK, main

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
    assert capsys.readouterr().out.startswith("usage: gun_probe2")


def test_the_module_entry_point_exits_with_the_probe_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.gun_probe2")
    sys.argv = ["gun_probe2"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.gun_probe2", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.gun_probe2"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: gun_probe2")


def test_the_timeline_records_turret_samples_and_skips_quiet_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One line set per turret-bearing sample -- roster, option rows, and
    the fresh ladder's verdict on the identical sample -- and nothing at
    all for samples with no base turret standing."""
    quiet = sample(entity(213, "commandCenter"), credits=4000)
    loud = sample(
        entity(213, "commandCenter"),
        entity(42, "c_turret_t1", x=100.0),
        credits=4000,
        options=(
            option(42, "c_turret_t2_gun", key="u_t2", price=1000),
            # Another unit's option: the timeline records turret rows only.
            option(213, "builder", key="u_builder", index=1, price=500),
        ),
    )
    timeline = tmp_path / "timeline.txt"
    peer = ScriptedPeer(lines(quiet, loud, quiet))
    with StubbedConnect(peer):
        assert main(_args(), timeline=timeline) == EXIT_OK
    recorded = timeline.read_text(encoding="utf-8")
    assert "turrets: 42(c=1,q=0)" in recorded
    assert "42 -> c_turret_t2_gun avail=1 price=1000" in recorded
    assert "'type_name': 'c_turret_t2_gun'" in recorded
    # The quiet samples wrote nothing: every recorded line names sample 1.
    assert all(line.startswith("s1 ") for line in recorded.splitlines())
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == f"timeline written to {timeline}"
