"""A batch of matches, described as the cluster's own campaign document.

Two load-bearing tests here. The members this package emits are handed to
hpc3's OWN decoder -- the whole reason the dependency exists, since a document
only its writer had checked would be refused at submission time, after the
game tree had already been staged. And every path a member carries is checked
for being absolute, because a relative one is accepted by every decoder, runs
against nothing on the cluster, and reports no error while a campaign
resubmits the whole batch forever.
"""

from __future__ import annotations

import shlex

import pytest
from hpc3.contracts.sweep import decode_sweep_member

from rw_bot.harness.campaign import (
    MATCH_MODULE,
    campaign_members,
    member_artifact,
    member_command,
)
from rw_bot.harness.match import MatchConfig
from rw_bot.harness.results_layout import result_path
from rw_bot.harness.sweep import SweepJob

PY = "/opt/env/bin/python"
ROOT = "/pub/wagnera3"
PROJECT = "rusted"
#: The staged tree this batch reads. A parameter of the document, not a
#: constant of the layout: an A/B is two documents differing exactly here.
PAYLOAD = "payload"
JOBS_FILE = "sweeps/demo.txt"
BATCH = "demo"
LOCKSTEP = 75
FASTFORWARD = 10

#: The match every member plays. Carried on the command rather than left to
#: the engine's ten-player default, because the map decides the opponent count
#: and is therefore the experiment.
MAP = "maps/skirmish/[p2]duel_lake.tmx"
MATCH = MatchConfig(map_path=MAP, opponents=1, difficulty=-2)


def _job(label: str = "attack", seed: int = 777) -> SweepJob:
    """Build one job of a batch.

    Args:
        label: Which arm.
        seed: What the engine's generator is pinned to.

    Returns:
        The job.
    """
    return SweepJob(label=label, seed=seed, doctrine="doctrines/a.doctrine", samples=1500)


def _command(job: SweepJob | None = None, batch: str = BATCH) -> str:
    """Build the command one member runs.

    Args:
        job: The job it plays. Defaults to the sample job.
        batch: The sweep it belongs to.

    Returns:
        The command.
    """
    return member_command(
        PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, batch, job or _job(), LOCKSTEP, FASTFORWARD, MATCH
    )


class TestWhereAMatchFilesItsResult:
    def test_the_relative_path_is_the_batchs_own_scorecard(self) -> None:
        assert result_path(BATCH, _job()) == "runs/sweeps/demo/attack-s777.txt"

    def test_the_cluster_path_hangs_it_off_the_projects_directory(self) -> None:
        """Under the project rather than beside it, because that is where
        ``hpc3`` already puts a project's scripts and logs."""
        assert member_artifact(ROOT, PROJECT, BATCH, _job()) == (
            "/pub/wagnera3/rusted/runs/sweeps/demo/attack-s777.txt"
        )

    def test_two_matches_of_one_arm_file_apart(self) -> None:
        """A member is done when ITS artifact exists, so two members sharing
        one would each read as done the moment the other finished."""
        assert result_path(BATCH, _job(seed=1)) != result_path(BATCH, _job(seed=2))

    def test_two_arms_file_apart(self) -> None:
        assert result_path(BATCH, _job("attack")) != result_path(BATCH, _job("defend"))


class TestEveryPathIsAbsolute:
    """The defect this closed. ``hpc3``'s campaign tests artifacts with
    ``[ -e <path> ]`` over SSH, whose working directory is ``$HOME``, and
    ``sbatch`` sets no working directory for the job. A relative path is
    refused by nothing, resolves against the wrong place, and makes a campaign
    resubmit its whole batch on every pass without ever converging."""

    def test_the_declared_artifact_is_absolute(self) -> None:
        assert member_artifact(ROOT, PROJECT, BATCH, _job()).startswith("/")

    def test_every_path_the_command_names_is_absolute(self) -> None:
        """Not only the artifact: the job file and the game tree are read by
        the match, and a relative one of those fails on the node instead of
        silently in the queue."""
        command = _command()
        for flag in ("--jobs", "--game", "--traces", "--result"):
            value = command.split(f"{flag} ")[1].split(" ")[0]
            assert value.startswith("/"), f"{flag} named a relative path {value!r}"

    def test_the_trace_root_is_named_because_the_measurement_lives_there(self) -> None:
        """The trace was composed inside the launcher against
        ``runs/traces`` until 2026-08-29 -- a path a compute node resolves
        against a home directory. A scorecard would still have been filed, so
        the campaign would have converged while the per-sample record, which
        is the entire measurement of a replication panel, went where nothing
        looked and nothing reported it."""
        assert f"--traces {ROOT}/{PROJECT}/runs/traces" in _command()

    def test_the_root_and_the_project_both_reach_the_paths(self) -> None:
        moved = member_command(
            PY,
            "/dfs6b/pub/other",
            "rw-second",
            PAYLOAD,
            JOBS_FILE,
            BATCH,
            _job(),
            LOCKSTEP,
            FASTFORWARD,
            MATCH,
        )
        assert "/dfs6b/pub/other/rw-second/" in moved
        assert ROOT not in moved


class TestTheMemberCommand:
    def test_it_runs_the_single_match_entry_point(self) -> None:
        """A whole batch's results directory is written by every member; a
        per-match scorecard is written by one, which is what a campaign needs."""
        assert f"-m {MATCH_MODULE}" in _command()

    def test_the_entry_point_is_one_the_wheel_actually_installs(self) -> None:
        """``scripts/`` is not packaged, so the ``scripts.match`` this used to
        name did not exist inside the image the members run in. Asserted on
        the module's real dotted path rather than on a string, so a rename
        moves the command with it."""
        assert MATCH_MODULE == "rw_bot.harness.campaign_match"

    def test_it_runs_the_images_own_interpreter(self) -> None:
        assert _command().startswith(PY)

    def test_it_names_the_one_match_it_plays(self) -> None:
        command = _command(_job("defend", 42))
        assert "--label defend" in command
        assert "--seed 42" in command

    def test_it_names_the_tree_to_play_from(self) -> None:
        """A compute node has no repository to be relative to, so the staged
        tree is named rather than assumed."""
        assert f"--game {ROOT}/{PROJECT}/game" in _command()

    def test_it_mentions_the_artifact_it_declares(self) -> None:
        """hpc3 refuses a member whose declared artifact its own command never
        names -- an index that publishes a path the run never wrote to is a
        confident wrong answer."""
        job = _job()
        assert member_artifact(ROOT, PROJECT, BATCH, job) in _command(job)

    def test_it_carries_the_batchs_lockstep(self) -> None:
        """Free-running, parallel matches sample at different game-times, so
        the lockstep is part of the experiment rather than a knob per node."""
        assert "--lockstep 75" in _command()

    def test_it_carries_the_batchs_fast_forward(self) -> None:
        """Pace is part of the regime the batch ran under: a member left to a
        default would run a pace the campaign document never stated, and a
        fast-forwarded batch would read as comparable to a realtime one."""
        assert "--fast-forward 10" in _command()


class TestTheMembers:
    def test_one_member_per_match_in_file_order(self) -> None:
        jobs = [_job("attack", 1), _job("attack", 2), _job("defend", 1)]
        members = campaign_members(
            PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, BATCH, jobs, LOCKSTEP, FASTFORWARD, MATCH
        )
        assert [member["suffix"] for member in members] == [
            "attack-s1",
            "attack-s2",
            "defend-s1",
        ]

    def test_every_member_declares_its_own_artifact(self) -> None:
        jobs = [_job("attack", 1), _job("attack", 2)]
        members = campaign_members(
            PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, BATCH, jobs, LOCKSTEP, FASTFORWARD, MATCH
        )
        artifacts = [member["artifact"] for member in members]
        assert len(set(artifacts)) == len(artifacts)

    def test_an_empty_batch_is_refused(self) -> None:
        """An empty campaign converges immediately and reports an experiment
        complete having played nothing."""
        with pytest.raises(ValueError, match="at least one member"):
            campaign_members(
                PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, BATCH, [], LOCKSTEP, FASTFORWARD, MATCH
            )

    def test_every_member_survives_hpc3s_own_decoder(self) -> None:
        """The reason this package depends on hpc3 rather than describing its
        document format a second time. A member that only this side had
        checked would be refused at submission, after staging."""
        jobs = [_job("attack", 1), _job("defend", 2)]
        for member in campaign_members(
            PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, BATCH, jobs, LOCKSTEP, FASTFORWARD, MATCH
        ):
            decoded = decode_sweep_member(
                {
                    "suffix": member["suffix"],
                    "command": member["command"],
                    "artifact": member["artifact"],
                }
            )
            assert decoded == member


class TestClonesAreNodeLocal:
    """Clones lived on BeeGFS until 2026-09-01, and many members booting at
    once crawled the engine's asset loading past the wrong-world guard's 60s
    deadline -- ten members across four batches halted with `after 60s the
    live world is null`, every one completing on an uncontended retry. The
    job's own $TMPDIR (local disk, measured 1.9 GB/s, removed with the job)
    is where a disposable per-member copy belongs."""

    def test_the_member_clones_into_the_jobs_own_scratch(self) -> None:
        """Unexpanded in the document, deliberately: the batch script's bash
        expands it on the node, after Slurm has provisioned the directory,
        and the script's `set -u` makes a node without TMPDIR fail loudly."""
        assert " --clones $TMPDIR/rw-clones " in _command()

    def test_no_clone_path_touches_the_shared_filesystem(self) -> None:
        command = _command()
        clones = command.split("--clones ")[1].split(" ")[0]
        assert not clones.startswith(ROOT)


class TestThePayloadIsAParameter:
    """Which staged tree a batch reads IS the experiment once two code
    versions are compared: an A/B is two documents whose members read their
    own trees side by side, and a hardwired payload directory made that
    impossible to state."""

    def test_the_jobs_and_the_tree_both_read_from_the_named_payload(self) -> None:
        command = _command()
        assert f"--jobs {ROOT}/{PROJECT}/payload/sweeps/demo.txt" in command
        assert f"--tree {ROOT}/{PROJECT}/payload" in command

    def test_a_different_payload_moves_both_reads_and_nothing_else(self) -> None:
        other = member_command(
            PY, ROOT, PROJECT, "payload-v7", JOBS_FILE, BATCH, _job(), LOCKSTEP, FASTFORWARD, MATCH
        )
        assert f"--jobs {ROOT}/{PROJECT}/payload-v7/sweeps/demo.txt" in other
        assert f"--tree {ROOT}/{PROJECT}/payload-v7" in other
        # The tree is the ONLY difference between the arms of an A/B; a
        # payload that leaked into any other path would confound it.
        assert other.replace("payload-v7", "payload") == _command()


class TestTheMatchIsCarried:
    """Which map is played decides how many opponents there are, so the map IS
    the experiment. A member that fell back to the engine's own ten-player
    free-for-all would run a different simulation from every batch this
    project has measured, and nothing in the document would say so."""

    def test_the_command_names_the_map(self) -> None:
        argv = shlex.split(_command())
        assert argv[argv.index("--map") + 1] == MAP

    def test_the_command_names_the_difficulty(self) -> None:
        assert "--difficulty -2" in _command()

    def test_the_opponent_count_is_not_passed_because_the_map_caps_it(self) -> None:
        """The engine caps teams by the map's own count, so a two-player map
        is a duel whatever is asked for. Passing it would be a second answer
        to a question the map already settles."""
        assert "--opponents" not in _command()

    def test_a_different_map_reaches_the_command(self) -> None:
        elsewhere = MatchConfig(
            map_path="maps/skirmish/[p2]big_island.tmx", opponents=1, difficulty=1
        )
        moved = member_command(
            PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, BATCH, _job(), LOCKSTEP, FASTFORWARD, elsewhere
        )
        argv = shlex.split(moved)
        assert argv[argv.index("--map") + 1] == "maps/skirmish/[p2]big_island.tmx"
        assert "--difficulty 1" in moved


class TestTheMapIsSafeInAShellCommand:
    """A member's command becomes a line in a batch script that bash runs, and
    these map names carry brackets, spaces and parentheses."""

    def test_a_bracketed_name_is_quoted(self) -> None:
        """``[p2]duel_lake.tmx`` survives unquoted only because the glob
        happens to match nothing and bash passes it through. That is luck,
        not correctness."""
        assert f"--map '{MAP}'" in _command()

    def test_a_name_with_spaces_and_parentheses_survives(self) -> None:
        """``[p2]Lake (2p).tmx`` is the name Steam ships the duel map under,
        and unquoted it is a bash SYNTAX ERROR rather than a wrong path."""
        steam = MatchConfig(map_path="maps/skirmish/[p2]Lake (2p).tmx", opponents=1, difficulty=-2)
        command = member_command(
            PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, BATCH, _job(), LOCKSTEP, FASTFORWARD, steam
        )
        assert "--map 'maps/skirmish/[p2]Lake (2p).tmx'" in command

    def test_the_quoted_command_survives_a_real_shell_split(self) -> None:
        """Asked of the shell's own parser rather than of a regular
        expression: what matters is what bash makes of the line."""
        steam = MatchConfig(map_path="maps/skirmish/[p2]Lake (2p).tmx", opponents=1, difficulty=-2)
        command = member_command(
            PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, BATCH, _job(), LOCKSTEP, FASTFORWARD, steam
        )
        argv = shlex.split(command)
        assert argv[argv.index("--map") + 1] == "maps/skirmish/[p2]Lake (2p).tmx"

    def test_every_member_of_a_batch_is_split_the_same_way(self) -> None:
        jobs = [_job("attack", 1), _job("defend", 2)]
        for member in campaign_members(
            PY, ROOT, PROJECT, PAYLOAD, JOBS_FILE, BATCH, jobs, LOCKSTEP, FASTFORWARD, MATCH
        ):
            argv = shlex.split(member["command"])
            assert argv[argv.index("--map") + 1] == MAP
            assert argv[argv.index("--result") + 1] == member["artifact"]
