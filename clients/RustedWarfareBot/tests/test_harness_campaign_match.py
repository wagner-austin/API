"""Playing exactly one match of a batch: the cluster's unit of work.

Driven end to end against the in-memory host, so the real job selection, the
real tree freeze and the real clone check run -- only the filesystem and the
game are fakes.

Every path here is an ABSOLUTE cluster path, because that is what a member is
handed. Composed through :mod:`rw_bot.harness.results_layout` rather than
written out, so a test cannot pass against a spelling the emitter does not
produce.
"""

from __future__ import annotations

import runpy

import pytest

from rw_bot.harness.campaign import member_artifact
from rw_bot.harness.campaign_match import (
    EXIT_INCOMPLETE,
    EXIT_OK,
    MatchCommandError,
    check_result_agrees,
    main,
    select_member,
)
from rw_bot.harness.clone import CLONE_PREFIX, DISPLAY_BASE, PLAY_PORT_BASE
from rw_bot.harness.results_layout import (
    PAYLOAD_DIR,
    TRACE_ROOT,
    clones_path,
    cluster_path,
)
from rw_bot.harness.runner import FROZEN_ENTRIES, TREE_MARKER
from rw_bot.harness.sweep import SweepError, SweepJob
from tests.harness_fakes import FakeHost

_ROOT = "/pub/wagnera3"
_PROJECT = "rusted"
_GAME = cluster_path(_ROOT, _PROJECT, "game")
_TRACES = cluster_path(_ROOT, _PROJECT, TRACE_ROOT)
_TREE = cluster_path(_ROOT, _PROJECT, PAYLOAD_DIR)

#: The map every member plays. Required rather than defaulted: falling back to
#: the engine's ten-player free-for-all would run a different simulation from
#: every batch this project has measured, with nothing saying which.
_MAP = "maps/skirmish/[p2]duel_lake.tmx"
_JOBS = cluster_path(_ROOT, _PROJECT, "sweeps/demo.txt")
_BATCH = "demo"

#: Where this batch's members make their clones. Given to a member rather
#: than resolved against its working directory, which on a compute node is a
#: HOME every node shares.
_CLONES = clones_path(_ROOT, _PROJECT, _BATCH)
_LINES = (
    "attack|1|doctrines/a.doctrine|1500",
    "attack|2|doctrines/a.doctrine|1500",
    "defend|1|doctrines/b.doctrine|1500",
)


def _job(label: str = "attack", seed: int = 1) -> SweepJob:
    """Build one job.

    Args:
        label: Which arm.
        seed: The seed.

    Returns:
        The job.
    """
    return SweepJob(label=label, seed=seed, doctrine="doctrines/a.doctrine", samples=1500)


def _result(label: str = "attack", seed: int = 1) -> str:
    """Return the absolute path the named match must write.

    Args:
        label: Which arm.
        seed: The seed.

    Returns:
        The artifact path, composed exactly as the campaign document composes
        it.
    """
    return member_artifact(_ROOT, _PROJECT, _BATCH, _job(label, seed))


def _argv(label: str = "attack", seed: int = 1) -> list[str]:
    """Build the command line a campaign member runs.

    Args:
        label: Which arm.
        seed: The seed.

    Returns:
        The arguments after the program name.
    """
    return [
        "--jobs",
        _JOBS,
        "--batch",
        _BATCH,
        "--label",
        label,
        "--seed",
        str(seed),
        "--lockstep",
        "75",
        "--fast-forward",
        "10",
        "--game",
        _GAME,
        "--tree",
        _TREE,
        "--traces",
        _TRACES,
        "--map",
        _MAP,
        "--difficulty",
        "-2",
        "--clones",
        _CLONES,
        "--result",
        _result(label, seed),
    ]


#: The batch's members, in the order its job file describes them -- which is
#: the order their leases come out in.
_MEMBERS = (("attack", 1), ("attack", 2), ("defend", 1))


def _clone(position: int) -> str:
    """Return the game copy the member at one position owns.

    Args:
        position: The member's place in the batch, from zero.

    Returns:
        The absolute clone directory, under the batch's own clone root.
    """
    return f"{_CLONES}/{CLONE_PREFIX}{position + 1}"


def _leases_of(label: str, seed: int) -> tuple[str, str, str]:
    """Run one member and report the three things its lease decides.

    Args:
        label: Which arm.
        seed: The seed.

    Returns:
        Its clone directory, its channel port and its X display, read off the
        launcher command the member actually issued.
    """
    with _planted() as host:
        main(_argv(label, seed))
    launch = next(argv for argv in host.commands if "--game-dir" in argv)
    return (
        launch[launch.index("--game-dir") + 1],
        launch[launch.index("--port") + 1],
        launch[launch.index("--display") + 1],
    )


def _planted() -> FakeHost:
    """Build a host holding the staged game and the batch's job file.

    Returns:
        The host.
    """
    host = FakeHost()
    host.plant_source(_GAME)
    host.files[_JOBS] = _LINES
    _plant_payload(host)
    return host


def _plant_payload(host: FakeHost) -> None:
    """Plant the frozen tree a member is staged with.

    A member does not freeze its own -- ``prepare_tree`` copies from
    repository-relative paths and a compute node has no repository -- so it
    checks what it was handed instead. Planted through the same constant the
    check reads, so a test cannot pass against a layout nothing produces.

    Args:
        host: The host to plant into.
    """
    host.dirs.add(_TREE)
    for name in FROZEN_ENTRIES:
        host.files[f"{_TREE}/{name}"] = ()
        host.dirs.add(f"{_TREE}/{name}")
    host.files[f"{_TREE}/{TREE_MARKER}"] = ("frozen",)


class TestSelectingTheMember:
    def test_the_one_job_named_is_returned_with_its_position(self) -> None:
        jobs = [_job("attack", 1), _job("attack", 2), _job("defend", 1)]
        position, job = select_member(jobs, "defend", 1)
        assert (position, job["label"]) == (2, "defend")

    def test_every_member_of_a_batch_gets_a_different_position(self) -> None:
        """The position IS the lease -- the clone directory, the channel port
        and the X display all come off it. Every member reading it out of the
        same job file is what makes it exclusive without anything being
        passed between them; all twenty-four leasing ordinal 1 is what put
        two nodes into one directory (55663569/55663571)."""
        jobs = [_job("attack", 1), _job("attack", 2), _job("defend", 1)]
        leases = [select_member(jobs, job["label"], job["seed"])[0] for job in jobs]
        assert leases == [0, 1, 2]

    def test_a_job_the_batch_does_not_describe_is_refused(self) -> None:
        """A member that plays nothing would report success having run no
        match, and the campaign would wait forever for an artifact nothing
        was ever going to write."""
        with pytest.raises(MatchCommandError) as caught:
            select_member([_job("attack", 1)], "defend", 1)
        assert caught.value.code == "RW-MATCH-001"

    def test_a_duplicated_job_is_refused(self) -> None:
        with pytest.raises(MatchCommandError) as caught:
            select_member([_job("attack", 1), _job("attack", 1)], "attack", 1)
        assert caught.value.code == "RW-MATCH-001"


class TestTheDeclaredResultPath:
    def test_the_absolute_path_this_match_writes_is_accepted(self) -> None:
        check_result_agrees(_result(), _BATCH, _job())

    def test_the_prefix_is_not_compared_because_it_cannot_be(self) -> None:
        """The match is handed an absolute path and does not know the cluster
        root or the project that prefixed it, so it checks the part that
        identifies the match and no more."""
        elsewhere = cluster_path("/dfs6b/pub/other", "rw", "runs/sweeps/demo/attack-s1.txt")
        check_result_agrees(elsewhere, _BATCH, _job())

    def test_a_path_for_another_seed_is_refused(self) -> None:
        """The failure worth catching: a seed edited in the command and not
        in the artifact. Both are below the prefix, so both are compared."""
        with pytest.raises(MatchCommandError) as caught:
            check_result_agrees(_result(seed=2), _BATCH, _job(seed=1))
        assert caught.value.code == "RW-MATCH-002"

    def test_a_path_for_another_batch_is_refused(self) -> None:
        with pytest.raises(MatchCommandError) as caught:
            check_result_agrees(member_artifact(_ROOT, _PROJECT, "other", _job()), _BATCH, _job())
        assert caught.value.code == "RW-MATCH-002"

    def test_a_path_this_match_does_not_write_is_refused(self) -> None:
        """The ledger would publish a path the run never wrote to, and a
        reader following it cannot tell whether the run failed, wrote
        elsewhere, or was never going to write at all."""
        with pytest.raises(MatchCommandError) as caught:
            check_result_agrees(f"{_ROOT}/rusted/runs/sweeps/demo/wrong.txt", _BATCH, _job())
        assert caught.value.code == "RW-MATCH-002"


class TestPlayingIt:
    def test_a_finished_match_reports_success(self) -> None:
        with _planted():
            assert main(_argv()) == EXIT_OK

    def test_the_scorecard_is_filed_where_the_member_declared(self) -> None:
        with _planted() as host:
            main(_argv())
        assert _result() in host.files

    def test_the_results_directory_keeps_its_posix_spelling(self) -> None:
        """Split as a POSIX path rather than with ``Path``, which resolves to
        the flavour of the interpreter that is running. On this suite's
        Windows that would put backslashes into the batch's own config and out
        again into every path composed from it."""
        with _planted() as host:
            main(_argv())
        assert "\\" not in "".join(host.files)

    def test_an_incomplete_staged_tree_stops_the_member(self) -> None:
        """Checked rather than built. The marker alone would not do: it
        certifies that every copy before it finished and says nothing about a
        source that was absent when the freeze ran -- the agent jar is
        exactly that case, and no compute node can rebuild it."""
        host = _planted()
        del host.files[f"{_TREE}/rw-agent.jar"]
        host.dirs.discard(f"{_TREE}/rw-agent.jar")
        with host, pytest.raises(SweepError) as caught:
            main(_argv())
        assert caught.value.code == "RW-SWEEP-006"
        assert "rw-agent.jar" in str(caught.value)

    def test_a_member_does_not_freeze_its_own_tree(self) -> None:
        """``prepare_tree`` copies from repository-relative paths and a node
        has no repository, so a freeze there reports success having copied
        nothing. The staged tree is used as given."""
        with _planted() as host:
            main(_argv())
        assert not any("tree frozen" in line for line in host.printed)

    def test_the_match_plays_out_of_the_staged_tree(self) -> None:
        """A member imports the snapshot frozen before submission, not a
        working tree -- there is none on a node -- so every member of a
        campaign runs the same code whenever it happens to start."""
        with _planted() as host:
            main(_argv())
        launches = [argv for argv in host.commands if "--tree" in argv]
        assert launches != []
        for argv in launches:
            assert argv[argv.index("--tree") + 1] == _TREE

    def test_a_match_that_prints_no_verdict_files_nothing(self) -> None:
        """The campaign reads that as still-missing and submits it again,
        which is correct: a blank filed as a measurement is the outcome to
        avoid."""
        host = _planted()
        host.transcripts[_clone(0)] = ("[play] the agent never opened port 27511",)
        with host:
            assert main(_argv()) == EXIT_INCOMPLETE
            assert _result() not in host.files

    def test_a_finished_member_releases_its_clone(self) -> None:
        """Success releases, failure retains. A clone is a per-member copy of
        the game with nothing in it a verdict needs; left behind, a finished
        batch's clones were 1.7 GB each and four batches deep before anyone
        deleted them by hand (2026-09-01)."""
        with _planted() as host:
            assert main(_argv()) == EXIT_OK
        assert _clone(0) in host.removed

    def test_a_failed_member_keeps_its_clone_because_the_wreckage_is_there(self) -> None:
        host = _planted()
        host.transcripts[_clone(0)] = ("[play] the agent never opened port 27511",)
        with host:
            assert main(_argv()) == EXIT_INCOMPLETE
        assert host.removed == []

    def test_each_member_clones_ports_and_displays_under_its_own_lease(self) -> None:
        """Every member used to lease ordinal 1. All of them aimed at one
        ``.game-w1``, resolved against the directory they were submitted from
        -- ``sbatch`` sets no working directory. The first began copying 307
        MB in, and the second nine seconds later on another node saw the
        directory already there, skipped the copy and died listing a maps
        directory that did not exist yet (jobs 55663569/55663571,
        2026-08-30). Two on ONE node would also have bound one port and
        started one X server twice."""
        leased = [_leases_of(label, seed) for label, seed in _MEMBERS]
        assert leased == [
            (_clone(position), str(PLAY_PORT_BASE + position + 1), str(DISPLAY_BASE + position + 1))
            for position in range(len(_MEMBERS))
        ]
        assert len({clone for clone, _, _ in leased}) == len(_MEMBERS)

    def test_a_missing_flag_is_refused(self) -> None:
        with _planted(), pytest.raises(ValueError, match="--result is required"):
            main(_argv()[:-2])

    def test_it_reads_the_process_arguments_when_given_none(self) -> None:
        with _planted() as host:
            host.argv = _argv()
            assert main(None) == EXIT_OK

    def test_the_module_guard_runs_main(self) -> None:
        """``python -m rw_bot.harness.campaign_match`` is what every campaign
        member runs -- an INSTALLED module, because ``scripts/`` is not in the
        wheel and the ``scripts.match`` this replaced could never have been
        imported inside the image."""
        with _planted() as host:
            host.argv = _argv()
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("rw_bot.harness.campaign_match", run_name="__main__")
            assert caught.value.code == EXIT_OK
