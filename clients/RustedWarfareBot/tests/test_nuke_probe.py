"""The nuke probe, driven end to end against a scripted game.

The chain under test is the finisher's: stockpile on the priced action,
launch on the free one with the point chosen by the planner, and judge the
strike by the blast circle emptying -- each step pinned to the exact wire
line it must produce, because the targeted ability is a new verb and this
probe is its first caller.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.nuke_probe import (
    EXIT_BAD_USAGE,
    EXIT_NO_LAUNCH,
    EXIT_OK,
    LAUNCH_RETRY_SAMPLES,
    main,
)

from tests.wire_fixtures import ScriptedPeer, StubbedConnect, entity, lines, option, sample

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CATALOGUE_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m0-probe" / "printunits.log"
_PLACEMENT_PATH = _PROJECT_ROOT / "wiki" / "sources" / "m11-pools" / "type-flags.ndjson"

_CENTRE = entity(213, "commandCenter")
_LAUNCHER = entity(500, "nukeLauncherC", x=60.0)
_NEAR = entity(401, "extractorT1", x=300.0)
# The far target is an upgraded tier deliberately: the third live run armed
# and never launched because the target filter matched the base name alone,
# and by launch time every extractor had upgraded past it
# (`runs/nuke-probe3.out`).
_FAR = entity(402, "extractorT3", x=1200.0)

_BUILD_NUKE = option(500, "", key="c_buildNuke", price=11000)
_LAUNCH_NUKE = option(500, "", key="c_launchNuke", index=1, price=0)
_LAUNCH_GATED = option(500, "", key="c_launchNuke", index=1, price=0, available=False)


def _args(opening: str = "1", watch: str | None = None) -> list[str]:
    argv = ["27200", str(_CATALOGUE_PATH), str(_PLACEMENT_PATH), opening]
    if watch is not None:
        argv.append(watch)
    return argv


@pytest.mark.parametrize("argv", [[], ["27200"], ["a", "b", "c", "d", "e", "f"]])
def test_a_bad_argument_count_prints_usage(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: nuke_probe")


def test_the_module_entry_point_exits_with_the_probe_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.nuke_probe")
    sys.argv = ["nuke_probe"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.nuke_probe", run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.modules["scripts.nuke_probe"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE
    assert capsys.readouterr().out.startswith("usage: nuke_probe")


def test_the_chain_arms_launches_at_the_far_extractor_and_confirms_the_strike(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole verb, in wire lines: the priced action is fired plain, the
    free action is fired AT the extractor farthest from the base, and the
    strike is confirmed the sample its blast circle reads empty."""
    arming = sample(
        _CENTRE,
        _LAUNCHER,
        _NEAR,
        _FAR,
        credits=12000,
        options=(_BUILD_NUKE, _LAUNCH_GATED),
    )
    ready = sample(
        _CENTRE,
        _LAUNCHER,
        _NEAR,
        _FAR,
        credits=1000,
        options=(_BUILD_NUKE, _LAUNCH_NUKE),
    )
    # The far extractor is gone; the near one stands well outside the blast.
    struck = sample(_CENTRE, _LAUNCHER, _NEAR, credits=1000, options=(_BUILD_NUKE,))
    peer = ScriptedPeer(lines(arming, arming, arming, ready, struck))
    with StubbedConnect(peer):
        assert main(_args("1", "3")) == EXIT_OK
    sent = [line for line in peer.sent if "ability" in line]
    assert sent == [
        '{"kind":"ability","unit_id":500,"key":"c_buildNuke"}',
        '{"kind":"ability_at","unit_id":500,"x":1200.0,"y":0.0,"key":"c_launchNuke"}',
    ]
    printed = capsys.readouterr().out.splitlines()
    assert printed[-7:] == [
        "s0 launcher: [('queued', 0, 1), ('c_buildNuke', 11000, 1), ('c_launchNuke', 0, 0)]",
        "s0 armed: 'c_buildNuke' price 11000",
        "s1 launcher: [('queued', 0, 1), ('c_buildNuke', 11000, 1), ('c_launchNuke', 0, 1)]",
        "s1 launched: 'c_launchNuke' at (1200, 0) -- extractor 402",
        "s2 launcher: [('queued', 0, 1), ('c_buildNuke', 11000, 1)]",
        "s2 inside blast: []",
        "verdict: the targeted point was cleared",
    ]


def test_an_unaffordable_stockpile_waits_and_the_watch_reports_the_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Credits below the action's price arm nothing; the run says how far it
    got rather than pretending a strike."""
    broke = sample(
        _CENTRE,
        _LAUNCHER,
        _FAR,
        credits=4000,
        options=(_BUILD_NUKE, _LAUNCH_GATED),
    )
    peer = ScriptedPeer(lines(broke, broke, broke))
    with StubbedConnect(peer):
        assert main(_args("1", "1")) == EXIT_NO_LAUNCH
    assert not [line for line in peer.sent if "ability" in line]
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "verdict: no confirmed strike (unarmed)"


def test_a_launcher_never_standing_reports_an_unarmed_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    quiet = sample(_CENTRE, _FAR, credits=50000)
    peer = ScriptedPeer(lines(quiet, quiet, quiet))
    with StubbedConnect(peer):
        assert main(_args("1", "1")) == EXIT_NO_LAUNCH
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "verdict: no confirmed strike (unarmed)"


def test_a_gated_launch_holds_and_reports_an_armed_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Armed but never offered the launch: the ammo is queued and the free
    action stays unavailable, so the probe holds rather than firing into a
    closed gate the agent would refuse."""
    arming = sample(
        _CENTRE,
        _LAUNCHER,
        _FAR,
        credits=12000,
        options=(_BUILD_NUKE, _LAUNCH_GATED),
    )
    peer = ScriptedPeer(lines(arming, arming, arming, arming))
    with StubbedConnect(peer):
        assert main(_args("1", "2")) == EXIT_NO_LAUNCH
    sent = [line for line in peer.sent if "ability" in line]
    assert sent == ['{"kind":"ability","unit_id":500,"key":"c_buildNuke"}']
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "verdict: no confirmed strike (armed)"


def test_a_launch_with_no_extractor_standing_holds_for_a_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The launch needs a point worth proving something at; with no owned
    extractor to aim at, the probe holds armed rather than firing blind."""
    bare = sample(
        _CENTRE,
        _LAUNCHER,
        credits=12000,
        options=(_BUILD_NUKE, _LAUNCH_NUKE),
    )
    peer = ScriptedPeer(lines(bare, bare, bare, bare))
    with StubbedConnect(peer):
        assert main(_args("1", "2")) == EXIT_NO_LAUNCH
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "verdict: no confirmed strike (armed)"


def test_a_blast_circle_that_never_empties_reports_the_launched_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A launch the world never confirms is reported as exactly that -- the
    verb was dispatched and the observable did not follow, which is the
    reading that would send the investigation to the agent log."""
    ready = sample(
        _CENTRE,
        _LAUNCHER,
        _NEAR,
        _FAR,
        credits=12000,
        options=(_BUILD_NUKE, _LAUNCH_NUKE),
    )
    peer = ScriptedPeer(lines(ready, ready, ready, ready, ready))
    with StubbedConnect(peer):
        assert main(_args("1", "3")) == EXIT_NO_LAUNCH
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "verdict: no confirmed strike (launched, blast never cleared)"


def test_an_unanswered_launch_is_refired_at_the_same_point(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The launch flag does not carry the ammo gate -- the row reads
    available at zero ammo (`runs/nuke-probe3.out`) -- so a launch the world
    never answers is fired again at the same point once the retry window
    passes, rather than trusting a dispatch the engine may have dropped."""
    ready = sample(
        _CENTRE,
        _LAUNCHER,
        _NEAR,
        _FAR,
        credits=12000,
        options=(_BUILD_NUKE, _LAUNCH_NUKE),
    )
    watch = LAUNCH_RETRY_SAMPLES + 3
    peer = ScriptedPeer(lines(*(ready for _ in range(watch + 2))))
    with StubbedConnect(peer):
        assert main(_args("1", str(watch))) == EXIT_NO_LAUNCH
    targeted = [line for line in peer.sent if '"kind":"ability_at"' in line]
    assert targeted == [
        '{"kind":"ability_at","unit_id":500,"x":1200.0,"y":0.0,"key":"c_launchNuke"}',
        '{"kind":"ability_at","unit_id":500,"x":1200.0,"y":0.0,"key":"c_launchNuke"}',
    ]
    printed = capsys.readouterr().out.splitlines()
    assert printed[-2] == (f"s{LAUNCH_RETRY_SAMPLES + 1} relaunched: 'c_launchNuke' at (1200, 0)")


def test_a_gated_launch_row_is_not_refired_into(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The retry window passing changes nothing while the engine gates the
    row: a refire into a closed gate is the silent drop the retry exists to
    recover from, not a second attempt."""
    ready = sample(
        _CENTRE,
        _LAUNCHER,
        _NEAR,
        _FAR,
        credits=12000,
        options=(_BUILD_NUKE, _LAUNCH_NUKE),
    )
    gated = sample(
        _CENTRE,
        _LAUNCHER,
        _NEAR,
        _FAR,
        credits=12000,
        options=(_BUILD_NUKE, _LAUNCH_GATED),
    )
    watch = LAUNCH_RETRY_SAMPLES + 3
    worlds = [ready, ready, ready, ready] + [gated] * watch
    peer = ScriptedPeer(lines(*worlds))
    with StubbedConnect(peer):
        assert main(_args("1", str(watch))) == EXIT_NO_LAUNCH
    targeted = [line for line in peer.sent if '"kind":"ability_at"' in line]
    assert targeted == [
        '{"kind":"ability_at","unit_id":500,"x":1200.0,"y":0.0,"key":"c_launchNuke"}'
    ]
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "verdict: no confirmed strike (launched, blast never cleared)"
