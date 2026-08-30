"""A batch of matches, described as the cluster's own campaign document.

WHY A CAMPAIGN AND NOT A SWEEP. On this workstation a batch is one process
playing twelve matches across twelve workers, and resuming it is re-issuing
the command. On a cluster the twelve are twelve SCHEDULED jobs, any of which
can be preempted, and "which ones still need playing" stops being something a
single process knows. ``hpc3-campaign`` answers exactly that question -- it
computes the gap between the members declared and the artifacts that exist,
and submits only the difference -- so a batch becomes a campaign rather than
growing its own resume machinery.

ONE MEMBER PER MATCH, and the granularity is forced rather than chosen. A
member is done when its artifact exists AND a job that declared it completed,
so the artifact has to be something exactly one member writes. A whole batch's
results directory is written by all of them; the per-match scorecard is
written by one. The unit also has to sit under hpc3's sixty-minute preemption
threshold or it needs checkpointing, and a match is about twenty minutes --
the scorecard IS its checkpoint, and a lost one costs one match.

EVERY PATH A MEMBER CARRIES IS ABSOLUTE. That is not tidiness. ``hpc3``'s
campaign asks the cluster which artifacts exist by running ``[ -e <path> ]``
over SSH, whose working directory is a login shell's ``$HOME``; and ``sbatch``
sets no working directory for the job itself. A relative artifact therefore
tests absent forever, so every member reads as still-missing and the campaign
resubmits the entire batch on every pass without ever converging -- silently,
because nothing anywhere fails. The composition lives in
:mod:`rw_bot.harness.results_layout`, which is also what the match itself
checks its own result against.

THE SCHEMA IS HPC3'S, IMPORTED. This module builds
:class:`~hpc3.contracts.sweep.SweepMember` values rather than dictionaries
shaped like them, so the members a batch emits carry the type the cluster's
own decoder will read them back as. A second description of the format here
would be a fork whose first divergence is a document the submitter refuses
after the game tree has already been staged.

WHAT THIS DELIBERATELY DOES NOT BUILD is the template the members hang off.
``hpc3`` resolves that from its WORKSPACE -- the partition, the wall clock and
the image come from the project's declared defaults, which live on the machine
that submits and not in this repository. A document written here therefore
carries only what a batch can say for itself, and is validated when it is
submitted, against the workspace that is the only thing able to judge it.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence

from hpc3.contracts.sweep import SweepMember

from rw_bot.harness import campaign_match
from rw_bot.harness.match import MatchConfig
from rw_bot.harness.results_layout import (
    GAME_DIR,
    PAYLOAD_DIR,
    TRACE_ROOT,
    clones_path,
    cluster_path,
    result_path,
)
from rw_bot.harness.sweep import SweepJob, job_name

#: The module a member's command runs. One match, one job, one artifact.
#:
#: Read off the module rather than written as a string, so a rename moves the
#: command with it. The value it must have is an INSTALLED one: ``scripts/``
#: is not in the wheel, so the ``scripts.match`` this used to name did not
#: exist inside the image the members run in.
MATCH_MODULE = campaign_match.__name__


def member_command(
    interpreter: str,
    root: str,
    project: str,
    jobs_file: str,
    batch: str,
    job: SweepJob,
    lockstep: int,
    match: MatchConfig,
) -> str:
    """Return the command one member runs.

    Args:
        interpreter: The Python the match runs under, inside the image.
        root: The cluster root, from the hpc3 workspace.
        project: The hpc3 project this batch belongs to.
        jobs_file: The batch's job file, relative to the repository root.
        batch: The sweep this job belongs to.
        job: The job this member plays.
        lockstep: Engine frames between samples.
        match: Which match every member of this batch plays. Carried on every
            member rather than left to the engine's default, because the map
            decides the opponent count and therefore IS the experiment -- a
            member that fell back would run the ten-player free-for-all while
            every workstation batch ran the duel, and nothing would say so.
            The opponent count is not passed: the engine caps it by the map's
            own team count, so a two-player map is a duel regardless.

            Its path is SHELL-QUOTED, and that is not caution. A command
            becomes a line in a batch script that bash runs, and map names
            here carry brackets, spaces and parentheses:
            ``[p2]duel_lake.tmx`` survives unquoted only because the glob
            matches nothing and bash passes it through, while
            ``[p2]Lake (2p).tmx`` -- the name Steam ships that one under --
            is a syntax error. :class:`~rw_bot.harness.match.MatchConfig`
            says so in its own docstring, and this rendered it into a shell
            string anyway until it was quoted here.

    Returns:
        The command, with every path made absolute against the project's
        cluster directory and the result path named so the artifact this
        member declares is one its own command mentions.
    """
    payload = cluster_path(root, project, PAYLOAD_DIR)
    return (
        f"{interpreter} -m {MATCH_MODULE}"
        # Read out of the staged payload, not out of a repository. The job
        # file names which arms and seeds the batch played, so it is as much
        # the experiment as the doctrines beside it, and it travels with them.
        f" --jobs {payload}/{jobs_file}"
        f" --batch {batch}"
        f" --label {job['label']}"
        f" --seed {job['seed']}"
        f" --lockstep {lockstep}"
        f" --game {cluster_path(root, project, GAME_DIR)}"
        f" --tree {payload}"
        f" --traces {cluster_path(root, project, TRACE_ROOT)}"
        f" --map {shlex.quote(match['map_path'])}"
        f" --difficulty {match['difficulty']}"
        # Where this member's copy of the game is made. Given, because a
        # clone name is relative and `sbatch` sets no working directory --
        # which aimed the whole batch at one directory in the project's
        # script directory and killed the second member to reach it.
        f" --clones {clones_path(root, project, batch)}"
        f" --result {member_artifact(root, project, batch, job)}"
    )


def member_artifact(root: str, project: str, batch: str, job: SweepJob) -> str:
    """Return the file one member is entitled to have written.

    Args:
        root: The cluster root, from the hpc3 workspace.
        project: The hpc3 project this batch belongs to.
        batch: The sweep this job belongs to.
        job: The job this member plays.

    Returns:
        The absolute cluster path of the match's scorecard.
    """
    return cluster_path(root, project, result_path(batch, job))


def campaign_members(
    interpreter: str,
    root: str,
    project: str,
    jobs_file: str,
    batch: str,
    jobs: Sequence[SweepJob],
    lockstep: int,
    match: MatchConfig,
) -> list[SweepMember]:
    """Turn a batch's jobs into the members of a campaign.

    Args:
        interpreter: The Python each match runs under.
        root: The cluster root, from the hpc3 workspace.
        project: The hpc3 project this batch belongs to.
        jobs_file: The batch's job file, relative to the repository root.
        batch: The sweep's name.
        jobs: Every match the file describes.
        lockstep: Engine frames between samples.
        match: Which match every member plays.

    Returns:
        One member per match, in file order.

    Raises:
        ValueError: When the batch has no jobs. An empty campaign converges
            immediately and reports an experiment complete having played
            nothing, which is the one answer nobody wants from a batch.

    Note:
        Two members sharing a suffix is refused by
        :func:`~hpc3.contracts.sweep.decode_sweep_spec`, which owns that rule
        because it owns the document. Repeating the check here would be a
        second copy of it, and the second copy is the one that goes stale.
    """
    if not jobs:
        raise ValueError(
            "a campaign needs at least one member: an empty one converges immediately "
            "and reports an experiment complete having played nothing"
        )
    return [
        SweepMember(
            suffix=job_name(job),
            command=member_command(
                interpreter, root, project, jobs_file, batch, job, lockstep, match
            ),
            artifact=member_artifact(root, project, batch, job),
        )
        for job in jobs
    ]


__all__ = [
    "MATCH_MODULE",
    "campaign_members",
    "member_artifact",
    "member_command",
]
