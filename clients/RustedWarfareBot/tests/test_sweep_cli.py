"""The sweep entry point, driven end to end against an in-memory host.

The pool really runs, the partition really partitions, and the results are
really filed -- only the filesystem and the game are fakes.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.sweep import EXIT_BAD_USAGE, EXIT_INCOMPLETE, EXIT_OK, main

from rw_bot.harness.sweep import SweepError
from tests.harness_fakes import FakeHost

_JOBS = "sweeps/demo.txt"


def _plant(host: FakeHost, *lines: str) -> None:
    host.plant_source(".game")
    host.write_text_lines(Path(_JOBS), lines)


def _results(host: FakeHost) -> list[str]:
    prefix = "runs/sweeps/demo/"
    return sorted(
        key[len(prefix) :]
        for key in host.files
        # Immediate children only, which is what the reader sees: it lists the
        # directory rather than walking it. Matching everything beneath used
        # to be harmless and stopped being so when the frozen tree started
        # carrying the batch's own job file -- ``.tree/sweeps/demo.txt`` then
        # read as a thirteenth result.
        if key.startswith(prefix) and key.endswith(".txt") and "/" not in key[len(prefix) :]
    )


def _flag(argv: tuple[str, ...], name: str) -> str | None:
    """Return the value a launch command gives one flag.

    Args:
        argv: The command a sweep composed.
        name: The flag to read.

    Returns:
        Its value, or None when the flag is absent. Absence is a real answer:
        an option a frozen tree predates must not be mentioned at all, so
        "off" and "not given" are the same thing on the command line.
    """
    if name not in argv:
        return None
    return argv[argv.index(name) + 1]


@pytest.mark.parametrize("args", [[], ["one"], ["a", "b", "c", "d", "e"]])
def test_a_bad_argument_count_prints_usage(args: list[str]) -> None:
    with FakeHost() as host:
        assert main(args) == EXIT_BAD_USAGE
        assert any(line.startswith("usage: sweep") for line in host.printed)


def test_a_duel_is_asked_for_by_map_and_difficulty() -> None:
    """The whole batch plays one match setup, and the map decides the opponent
    count -- the engine caps teams by the map's own, so a two-player map is a
    duel ([[policy-determinism]]).

    Every measurement before this ran against the engine's hardcoded
    ten-player free-for-all, which nobody chose.
    """
    with FakeHost() as host:
        _plant(host, "duel|1|doctrines/default.doctrine|1500")
        assert main([_JOBS, "demo", "1", "75", "maps/skirmish/[p2]duel_lake.tmx", "-2"]) == EXIT_OK
        assert _flag(host.commands[0], "--map") == "maps/skirmish/[p2]duel_lake.tmx"
        assert _flag(host.commands[0], "--difficulty") == "-2"
        # Named by its handicap, because "difficulty -2" says nothing about
        # what it does and "0.4x AI income" says all of it.
        assert any("0.4x AI income" in line for line in host.printed)


def test_a_pinned_batch_passes_the_delta_to_every_match() -> None:
    """The seventh positional: a constant frame delta for the whole batch.

    Batch-level like the match, because a pinned and an unpinned run of one
    seed are different simulations ([[policy-determinism]]). Unpinned batches
    must stay silent -- a tree frozen before the option existed runs an agent
    that rejects the unknown key.
    """
    with FakeHost() as host:
        _plant(host, "duel|1|doctrines/default.doctrine|1500")
        code = main([_JOBS, "demo", "1", "75", "maps/skirmish/[p2]duel_lake.tmx", "1", "3"])
        assert code == EXIT_OK
        assert _flag(host.commands[0], "--pin-delta") == "3"
        host.commands.clear()
        del host.files["runs/sweeps/demo/duel-s1.txt"]

        assert main([_JOBS, "demo", "1", "75", "maps/skirmish/[p2]duel_lake.tmx", "1"]) == EXIT_OK
        assert _flag(host.commands[0], "--pin-delta") is None


def test_a_fast_batch_passes_the_multiple_to_every_match() -> None:
    """The eighth positional: the gym knob, certified bit-exact against
    realtime at 10x (log 2026-08-06). A batch that omits it stays silent so
    trees frozen before the option existed keep running."""
    with FakeHost() as host:
        _plant(host, "duel|1|doctrines/default.doctrine|1500")
        code = main([_JOBS, "demo", "1", "75", "maps/skirmish/[p2]duel_lake.tmx", "1", "3", "10"])
        assert code == EXIT_OK
        assert _flag(host.commands[0], "--fast-forward") == "10"
        assert _flag(host.commands[0], "--pin-delta") == "3"
        host.commands.clear()
        del host.files["runs/sweeps/demo/duel-s1.txt"]

        code = main([_JOBS, "demo", "1", "75", "maps/skirmish/[p2]duel_lake.tmx", "1", "3"])
        assert code == EXIT_OK
        assert _flag(host.commands[0], "--fast-forward") is None


def test_every_match_in_the_file_is_played_once() -> None:
    with FakeHost() as host:
        _plant(
            host,
            "tank|1|doctrines/default.doctrine|1500",
            "tank|2|doctrines/default.doctrine|1500",
            "arty|1|doctrines/arty.doctrine|1500",
        )
        assert main([_JOBS, "demo", "2"]) == EXIT_OK
        assert _results(host) == ["arty-s1.txt", "tank-s1.txt", "tank-s2.txt"]
        assert len(host.commands) == 3


def test_a_second_run_replays_only_what_is_missing() -> None:
    """This is the whole of resumability, and it is why a batch is never a
    single unit of work.
    """
    with FakeHost() as host:
        _plant(
            host,
            "tank|1|doctrines/default.doctrine|1500",
            "tank|2|doctrines/default.doctrine|1500",
        )
        assert main([_JOBS, "demo", "2"]) == EXIT_OK
        host.commands.clear()
        del host.files["runs/sweeps/demo/tank-s2.txt"]

        assert main([_JOBS, "demo", "2"]) == EXIT_OK
        assert len(host.commands) == 1
        assert _flag(host.commands[0], "--seed") == "2"
        assert any("1 already played" in line for line in host.printed)


def test_a_batch_that_could_not_finish_reports_it() -> None:
    with FakeHost(transcripts={".game-w1": ("[play] game stopped",)}) as host:
        _plant(host, "tank|1|doctrines/default.doctrine|1500")
        assert main([_JOBS, "demo", "1"]) == EXIT_INCOMPLETE
        assert _results(host) == []
        assert "runs/sweeps/demo/tank-s1.partial" in host.files


def test_the_pool_never_exceeds_the_number_of_matches() -> None:
    """A batch of one should not copy the game four times to leave three idle."""
    with FakeHost() as host:
        _plant(host, "tank|1|doctrines/default.doctrine|1500")
        assert main([_JOBS, "demo", "4"]) == EXIT_OK
        assert host.path_exists(Path(".game-w1"))
        assert not host.path_exists(Path(".game-w2"))
        assert any("over 1 workers" in line for line in host.printed)


def test_every_match_is_locked_to_the_tick_by_default() -> None:
    """Free running, parallel matches under CPU contention sample at different
    game-times, so running a sweep in parallel would change its results.
    """
    with FakeHost() as host:
        _plant(host, "tank|1|doctrines/default.doctrine|1500")
        assert main([_JOBS, "demo", "1"]) == EXIT_OK
        assert _flag(host.commands[0], "--lockstep") == "75"


def test_the_lockstep_is_an_argument_so_an_arm_can_change_it() -> None:
    with FakeHost() as host:
        _plant(host, "tank|1|doctrines/default.doctrine|1500")
        assert main([_JOBS, "demo", "1", "40"]) == EXIT_OK
        assert _flag(host.commands[0], "--lockstep") == "40"


def test_the_worker_count_defaults_when_not_given() -> None:
    with FakeHost() as host:
        _plant(host, *[f"tank|{n}|doctrines/default.doctrine|1500" for n in range(6)])
        assert main([_JOBS, "demo"]) == EXIT_OK
        assert any("over 4 workers" in line for line in host.printed)


def test_a_malformed_job_file_stops_the_batch_rather_than_playing_part_of_it() -> None:
    with FakeHost() as host:
        _plant(
            host,
            "tank|1|doctrines/default.doctrine|1500",
            "arty|nonsense|doctrines/arty.doctrine|1500",
        )
        with pytest.raises(SweepError) as caught:
            main([_JOBS, "demo", "2"])
        assert caught.value.code == "RW-SWEEP-002"
        assert host.commands == []


def test_a_batch_with_nothing_outstanding_starts_no_pool_at_all() -> None:
    """Re-running a finished sweep should cost nothing, not spin up workers to
    discover they have no work.
    """
    with FakeHost() as host:
        _plant(host, "tank|1|doctrines/default.doctrine|1500")
        assert main([_JOBS, "demo", "1"]) == EXIT_OK
        host.commands.clear()

        assert main([_JOBS, "demo", "1"]) == EXIT_OK
        assert host.commands == []
        assert any("0 to go" in line for line in host.printed)


def test_the_module_entry_point_exits_with_the_batch_result() -> None:
    original_argv = sys.argv
    already_imported = sys.modules.pop("scripts.sweep")
    sys.argv = ["sweep"]
    try:
        with FakeHost() as host, pytest.raises(SystemExit) as caught:
            runpy.run_module("scripts.sweep", run_name="__main__")
        assert any(line.startswith("usage: sweep") for line in host.printed)
    finally:
        sys.argv = original_argv
        sys.modules["scripts.sweep"] = already_imported
    assert caught.value.code == EXIT_BAD_USAGE


def test_the_arguments_are_read_from_the_process_when_none_are_given() -> None:
    with FakeHost() as host:
        _plant(host, "tank|1|doctrines/default.doctrine|1500")
        host.argv = [_JOBS, "demo", "1"]
        assert main(None) == EXIT_OK
        assert _results(host) == ["tank-s1.txt"]
