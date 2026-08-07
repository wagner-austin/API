"""The opening compile: what the plan reads, waits for, and refuses.

The plan is only as good as the roster it compiled against, and the
exact-timing regime made the first samples untrustworthy: the engine's boot
sandbox can be served before the configured world goes live, frame counter
already pinned to zero. These pin the settle discipline -- wait for content,
wait out the sandbox, and refuse an empty world -- that keeps the compile
honest (log 2026-08-06: the sandbox-poisoned plan cost two 0/24 panels).
"""

from __future__ import annotations

import pytest
from scripts.play import DEFAULT_GOALS, EXIT_INCOMPLETE, EXIT_OK, main

from tests.play_fixtures import (
    BUILDER,
    CATALOGUE_PATH,
    EXPANDED,
    PLACEMENT_PATH,
    SANDBOX,
    sample_lines,
)
from tests.wire_fixtures import ScriptedPeer, StubbedConnect


def test_the_opening_settles_by_content_not_by_clock(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A held match world may be sampled before its units spawn.

    The world used to settle on 22 seconds of free-running wall clock, and
    runs began from worlds that already differed ([[policy-determinism]]).
    The roster is what plan expansion reads, so the roster is the condition:
    empty samples are acknowledged and the plan is expanded against the first
    observation that owns something finished.
    """
    built = [(300 + i, name) for i, name in enumerate(EXPANDED)]
    empty = sample_lines(1, 4000)
    populated = sample_lines(2, 9000, BUILDER, *built)
    # Two empty observations, one populated one the settle consumes, then the
    # loop's own two samples.
    peer = ScriptedPeer(empty + empty + populated * 12)
    with StubbedConnect(peer):
        assert main(["27200", str(CATALOGUE_PATH), str(PLACEMENT_PATH), "2"]) == EXIT_OK
    printed = capsys.readouterr().out.splitlines()
    # The plan was expanded against the populated world, not the empty one.
    assert printed[3] == (
        "plan:  extractorT1 -> extractorT1 -> extractorT1 -> c_tank -> c_tank -> c_tank -> c_tank"
    )


def test_a_world_that_never_populates_is_a_failed_start_not_a_slow_one() -> None:
    """The settle is bounded: a match whose units never spawn is broken, and
    expanding a plan against an empty world says so loudly rather than
    waiting forever.
    """
    from rw_bot.policy.expand import ExpansionError

    empty = sample_lines(1, 4000)
    peer = ScriptedPeer(empty * 60)
    with StubbedConnect(peer), pytest.raises(ExpansionError) as caught:
        main(["27200", str(CATALOGUE_PATH), str(PLACEMENT_PATH), "1"])
    assert caught.value.code == "RW-EXPAND-001"


def test_the_plan_waits_out_the_boot_sandbox_before_compiling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exact-timing regime can serve the engine's ten-player boot world
    before the configured match goes live, frame counter already pinned to
    zero. A plan compiled against that roster inserts no factory -- it
    believes it owns two -- and is dead at its first combat entry
    (log 2026-08-06: 0/24 at Very Hard AND at Hard). The swap is a roster
    sharing no identity with the sandbox's."""
    sandbox = sample_lines(1, 4000, *SANDBOX)
    duel = sample_lines(2, 4000, BUILDER)
    peer = ScriptedPeer(sandbox + duel * 4)
    with StubbedConnect(peer):
        # Incomplete is right: a bare builder cannot finish a nine-entry plan
        # in two samples. The compile is what is under test.
        assert main(["27200", str(CATALOGUE_PATH), str(PLACEMENT_PATH), "2"]) == EXIT_INCOMPLETE
    printed = capsys.readouterr().out.splitlines()
    assert printed[1] == "owned at compile (frame 2): builder"
    assert printed[3] == "plan:  " + " -> ".join(EXPANDED)


def test_a_rich_world_that_never_swaps_is_the_run_s_real_world(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default-map probes play the sandbox itself; for them the rich
    roster is the truth and compiling against it is correct. The wait is
    bounded so those runs pay a window, not the match."""
    sandbox = sample_lines(1, 4000, *SANDBOX)
    peer = ScriptedPeer(sandbox * 12)
    with StubbedConnect(peer):
        main(["27200", str(CATALOGUE_PATH), str(PLACEMENT_PATH), "1"])
    printed = capsys.readouterr().out.splitlines()
    assert printed[1].startswith("owned at compile (frame 1):")
    # The sandbox owns a factory, so expansion inserts nothing.
    assert printed[3] == "plan:  " + " -> ".join(DEFAULT_GOALS)
