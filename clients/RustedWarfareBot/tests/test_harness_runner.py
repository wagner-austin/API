"""Playing a batch: the service, driven against an in-memory host.

The real control flow runs here -- the real clone, the real completeness test,
the real partition. Only the filesystem and the game are fakes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rw_bot.harness.clone import PLAY_PORT_BASE, CloneError
from rw_bot.harness.match import MatchConfig
from rw_bot.harness.runner import (
    SweepConfig,
    SweepOutcome,
    decode_sweep_config,
    decode_sweep_outcome,
    encode_sweep_config,
    encode_sweep_outcome,
    outstanding,
    play_job,
    prepare_clone,
    prepare_tree,
    reset_volatile_files,
    run_worker,
)
from rw_bot.harness.sweep import SweepJob
from tests.harness_fakes import FakeHost

_SOURCE = ".game"


def _config(
    workers: int = 2,
    out_dir: str = "runs/sweeps/demo",
    match: MatchConfig | None = None,
    pin_delta: int = 0,
    fast_forward: int = 0,
) -> SweepConfig:
    return SweepConfig(
        out_dir=out_dir,
        workers=workers,
        lockstep=75,
        clone_prefix=".game-w",
        source_game_dir=_SOURCE,
        tree=f"{out_dir}/.tree",
        pin_delta=pin_delta,
        fast_forward=fast_forward,
        match=match,
    )


def _job(label: str = "tank", seed: int = 1) -> SweepJob:
    return SweepJob(
        label=label,
        seed=seed,
        doctrine="doctrines/default.doctrine",
        samples=1500,
    )


def test_the_batch_freezes_the_tree_its_matches_will_import() -> None:
    """Frozen once, at launch: the working tree is editable the moment the
    batch starts, and the batch records exactly what its matches ran."""
    with FakeHost() as host:
        host.dirs.add("src/rw_bot")
        host.dirs.add("scripts")
        host.dirs.add("doctrines")
        host.files["agent/build/rw-agent.jar"] = ()
        prepare_tree(_config())
        assert host.path_exists(Path("runs/sweeps/demo/.tree/src/rw_bot"))
        assert host.path_exists(Path("runs/sweeps/demo/.tree/scripts"))
        assert host.path_exists(Path("runs/sweeps/demo/.tree/doctrines"))
        assert host.path_exists(Path("runs/sweeps/demo/.tree/rw-agent.jar"))
        assert host.path_exists(Path("runs/sweeps/demo/.tree/.complete"))
        assert "[sweep] tree frozen at runs/sweeps/demo/.tree" in host.printed


def test_an_existing_frozen_tree_is_reused_never_refreshed() -> None:
    """Resuming a batch resumes its code: matches played after an interruption
    import the same tree as the ones played before it, whatever has happened
    to the working tree in between."""
    with FakeHost() as host:
        host.dirs.add("runs/sweeps/demo/.tree")
        host.files["runs/sweeps/demo/.tree/.complete"] = ("frozen",)
        host.dirs.add("src/rw_bot")
        prepare_tree(_config())
        assert not host.path_exists(Path("runs/sweeps/demo/.tree/src/rw_bot"))
        assert "[sweep] reusing the frozen tree at runs/sweeps/demo/.tree" in host.printed


def test_a_clone_is_a_copy_of_the_game_without_the_trees_it_rewrites() -> None:
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        name = prepare_clone(0, _config())
        assert name == ".game-w1"
        assert host.path_exists(Path(".game-w1/jvm64/bin/java.exe"))
        assert host.path_exists(Path(".game-w1/game-lib.jar"))
        # Copied from the source, not created: the source's saves tree is left
        # behind and the game rebuilds it on boot.
        assert not host.path_exists(Path(".game-w1/saves"))


def test_an_existing_clone_is_verified_rather_than_recopied() -> None:
    """A sweep is run several times while an experiment is refined, and
    re-copying 0.44 GB on each of those buys nothing.
    """
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        prepare_clone(0, _config())
        before = dict(host.files)
        host.files["marker"] = ("untouched",)
        prepare_clone(0, _config())
        assert {k: v for k, v in host.files.items() if k != "marker"} == before


def test_a_truncated_clone_is_refused_before_a_match_is_launched() -> None:
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        del host.files[f"{_SOURCE}/game-lib.jar"]
        with pytest.raises(CloneError) as caught:
            prepare_clone(0, _config())
        assert caught.value.code == "RW-CLONE-001"


def test_a_finished_match_is_filed_as_its_scorecard() -> None:
    with FakeHost() as host:
        assert play_job(_job(seed=42), ".game-w1", _config()) is True
        # The whole file, so the planner's chatter being absent is asserted
        # rather than merely unmentioned.
        assert host.files["runs/sweeps/demo/tank-s42.txt"] == (
            "### tank-s42",
            "verdict        survived (sample_limit)",
            "army           0 -> 9",
        )


def test_a_scorecard_states_the_match_it_played() -> None:
    """The batch name was the only record of map and difficulty, and a
    dataset built across batches cannot read names. The card states its own
    setup, label-padded like every report line."""
    match = MatchConfig(map_path="maps/skirmish/[p2]duel_lake.tmx", opponents=1, difficulty=2)
    with FakeHost() as host:
        assert play_job(_job(seed=42), ".game-w1", _config(match=match)) is True
        assert host.files["runs/sweeps/demo/tank-s42.txt"] == (
            "### tank-s42",
            "match          1 opponent(s) at difficulty 2 (1.8x AI income) "
            "on maps/skirmish/[p2]duel_lake.tmx",
            "verdict        survived (sample_limit)",
            "army           0 -> 9",
        )


def test_a_reused_clone_receives_maps_the_pinned_copy_gained() -> None:
    """Clone reuse is cheap and was silently stale: six maps added to the
    pinned copy after the clones were made never reached them, the engine's
    load failed with an alert nothing read, and every match on those maps
    played the boot sandbox instead -- the "seating anomaly", solved
    (log 2026-08-06). The sync is reported, never silent."""
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        host.dirs.add(f"{_SOURCE}/assets/maps")
        host.dirs.add(f"{_SOURCE}/assets/maps/skirmish")
        host.files[f"{_SOURCE}/assets/maps/skirmish/[p2]duel_lake.tmx"] = ("map",)
        host.files[f"{_SOURCE}/assets/maps/skirmish/[p2]lake_2p.tmx"] = ("map",)
        # An existing, otherwise-complete clone that predates lake_2p.
        host.plant_source(".game-w1")
        host.dirs.add(".game-w1/assets/maps")
        host.dirs.add(".game-w1/assets/maps/skirmish")
        host.files[".game-w1/assets/maps/skirmish/[p2]duel_lake.tmx"] = ("map",)

        assert prepare_clone(0, _config()) == ".game-w1"
        assert ".game-w1/assets/maps/skirmish/[p2]lake_2p.tmx" in host.files
        assert any("synced 1 map(s)" in line for line in host.printed)


def test_a_clone_with_every_map_syncs_nothing_and_says_nothing() -> None:
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        host.dirs.add(f"{_SOURCE}/assets/maps")
        host.dirs.add(f"{_SOURCE}/assets/maps/skirmish")
        host.files[f"{_SOURCE}/assets/maps/skirmish/[p2]duel_lake.tmx"] = ("map",)
        assert prepare_clone(0, _config()) == ".game-w1"
        assert not any("synced" in line for line in host.printed)


def test_every_match_starts_from_the_pinned_settings_not_the_last_match_s() -> None:
    """The game rewrites preferences.ini on each boot, so without this the
    second match a worker plays starts from the first one's leavings and two
    workers that have played different numbers of matches start from different
    settings. Measured, the drift is a main-menu counter and cannot reach a
    headless simulation -- but the guarantee an experiment needs is that the
    state does not differ, not that the difference is currently harmless.
    """
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        host.files[f"{_SOURCE}/preferences.ini"] = ("nextBackgroundMap:1",)
        host.files[".game-w1/preferences.ini"] = ("nextBackgroundMap:9",)

        play_job(_job(), ".game-w1", _config())

        assert host.files[".game-w1/preferences.ini"] == ("nextBackgroundMap:1",)


def test_the_settings_are_reset_before_the_match_rather_than_after() -> None:
    """Resetting afterwards would leave the very first match of a batch running
    on whatever the previous batch left behind.
    """
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        host.files[f"{_SOURCE}/preferences.ini"] = ("pinned",)
        host.files[".game-w1/preferences.ini"] = ("stale",)
        reset_volatile_files(".game-w1", _config())
        assert host.files[".game-w1/preferences.ini"] == ("pinned",)


def test_a_match_that_never_reported_a_verdict_is_not_filed_as_a_result() -> None:
    """Filing it would record a blank as though it were a measurement, and the
    job would never be replayed.
    """
    with FakeHost(transcripts={".game-w1": ("goals: c_tank", "[play] game stopped")}) as host:
        assert play_job(_job(seed=7), ".game-w1", _config()) is False
        assert "runs/sweeps/demo/tank-s7.txt" not in host.files
        assert "runs/sweeps/demo/tank-s7.partial" in host.files


def test_a_worker_plays_its_share_and_no_other() -> None:
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        jobs = [_job(seed=n) for n in range(4)]
        assert run_worker(jobs, 0, _config(workers=2)) == 2
        played = sorted(
            key
            for key in host.files
            if key.startswith("runs/sweeps/demo/") and key.endswith(".txt")
        )
        assert played == ["runs/sweeps/demo/tank-s0.txt", "runs/sweeps/demo/tank-s2.txt"]


def test_a_worker_with_no_share_never_pays_for_a_clone() -> None:
    """A batch smaller than the pool should not copy the game to leave the copy
    idle.
    """
    with FakeHost() as host:
        host.plant_source(_SOURCE)
        assert run_worker([_job()], 1, _config(workers=2)) == 0
        assert not host.path_exists(Path(".game-w2"))


def test_a_failed_match_is_not_counted_as_played() -> None:
    with FakeHost(transcripts={".game-w1": ("[play] game stopped",)}) as host:
        host.plant_source(_SOURCE)
        assert run_worker([_job()], 0, _config(workers=1)) == 0


def test_a_match_with_a_result_is_not_replayed() -> None:
    """Resumability is the result files and nothing else, so nothing can
    disagree with them.
    """
    with FakeHost() as host:
        host.write_text_lines(Path("runs/sweeps/demo/tank-s1.txt"), ("### tank-s1",))
        todo = outstanding([_job(seed=1), _job(seed=2)], Path("runs/sweeps/demo"))
        assert [job["seed"] for job in todo] == [2]


def test_a_partial_transcript_does_not_count_as_a_result() -> None:
    with FakeHost() as host:
        host.write_text_lines(Path("runs/sweeps/demo/tank-s1.partial"), ("### tank-s1 FAILED",))
        todo = outstanding([_job(seed=1)], Path("runs/sweeps/demo"))
        assert [job["seed"] for job in todo] == [1]


def test_a_configuration_round_trips_through_its_payload() -> None:
    assert decode_sweep_config(encode_sweep_config(_config())) == _config()
    fast = _config(fast_forward=10)
    assert decode_sweep_config(encode_sweep_config(fast)) == fast


def test_an_outcome_round_trips_through_its_payload() -> None:
    outcome = SweepOutcome(total=6, already=2, played=4)
    assert decode_sweep_outcome(encode_sweep_outcome(outcome)) == outcome


def test_a_cloned_match_plays_on_the_port_its_lease_owns() -> None:
    """play_job derives the port from the clone ordinal, so two concurrent
    matches can no more share a port than a directory."""
    with FakeHost() as host:
        assert play_job(_job(seed=42), ".game-w3", _config()) is True
        assert f"PLAY_PORT={PLAY_PORT_BASE + 3}" in host.commands[0]
