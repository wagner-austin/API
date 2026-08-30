"""Where a batch's files are, on a workstation and on the cluster.

ONE LAYOUT, TWO ROOTS. A batch files the same tree of results whether it plays
here or on a compute node; what differs is only what that tree hangs off. On
this workstation it hangs off the repository, so the paths are relative and a
process started in the repository finds them. On the cluster nothing sets a
working directory -- ``sbatch`` inherits whatever the submitting shell had --
so the same paths must be absolute or they resolve against a home directory
nobody put anything in.

THE ABSOLUTE HALF IS NOT OPTIONAL AND ITS ABSENCE IS SILENT. ``hpc3``'s
campaign asks the cluster which artifacts exist by running ``[ -e <path> ]``
over SSH, from a login shell whose working directory is ``$HOME``. A relative
artifact tests ABSENT there every time, so every member reads as still-missing,
and a campaign that is meant to converge resubmits the whole batch on every
pass. Nothing errors; the queue just fills again.

WHY THIS MODULE EXISTS RATHER THAN THE PATHS LIVING WITH THEIR USERS. The
member's command and the member's declared artifact must name the same file --
that is the invariant ``hpc3.contracts.job.require_artifact_in_command``
checks -- and the match that runs must write it. Three callers, so the
composition is here and none of them spells it. It also stopped the suffix
being declared twice, in :mod:`rw_bot.harness.campaign` and
:mod:`rw_bot.harness.records`, which is how one of them would eventually have
become ``.result``.
"""

from __future__ import annotations

from hpc3.contracts.layout import project_dir

from rw_bot.harness.sweep import SweepJob, job_name

#: Where a batch's results are filed, relative to the root. Forward-slashed
#: because the same string is read on the cluster.
SWEEP_ROOT = "runs/sweeps"

#: Suffix of a filed match result.
RESULT_SUFFIX = ".txt"

#: Where a match's trace lives, relative to the trace root.
TRACE_SUFFIX = ".ndjson"

#: Where a batch's per-sample traces are filed, relative to the root.
#:
#: A SEPARATE root from the results rather than a subdirectory of them,
#: because that is where they have always gone and the analyser reads them
#: from there. What changed is that it is now passed to a match instead of
#: assumed: on a compute node ``runs/traces`` resolves against a home
#: directory, so the trace -- which for a replication panel IS the whole
#: measurement -- landed somewhere nothing would look and nothing reported it.
TRACE_ROOT = "runs/traces"

#: Where the engine's and the agent's own logs go, under a batch's results.
#:
#: The deep-debug layer, kept per batch rather than on a shared floor: one of
#: these named the placeholder bug outright, and the shared floor overwrote
#: them across batches and replays (log 2026-07-31).
LOG_DIR = "logs"
LOG_SUFFIX = ".log"

#: The staged game tree, relative to the root. On this workstation the pinned
#: directory is :data:`PINNED_GAME_DIR`; on the cluster it is a staged copy
#: under the project's own directory, so the name is given rather than
#: assumed.
GAME_DIR = "game"

#: The staged frozen tree, relative to the root.
#:
#: A cluster member does not freeze its own. ``prepare_tree`` copies from
#: repository-relative paths and a compute node has no repository, so a freeze
#: there would report success having copied nothing -- and the agent jar it
#: carries cannot be rebuilt on a node at all, because the Linux depot ships a
#: JRE with no compiler. The freeze happens before submission and is staged,
#: which also makes it a digest-pinned artifact rather than a directory each
#: node assembled for itself.
PAYLOAD_DIR = "payload"

#: Where a batch's per-member game clones live, relative to the root.
#:
#: Given to a cluster member rather than left to resolve against its working
#: directory, and both halves of that were measured on one submission.
#:
#: A clone name is relative -- ``.game-w1`` -- and ``sbatch`` sets no working
#: directory, so a member resolves it against the directory it was SUBMITTED
#: from. ``hpc3`` submits with ``cd <script_dir> && sbatch``, and that is one
#: directory on ``/pub`` for the whole project. Two members nine seconds
#: apart on different nodes therefore aimed at ONE clone: the first made it
#: and began copying 307 MB in, the second saw it already existed, skipped
#: the copy it thought was done, and died listing
#: ``.game-w1/assets/maps/skirmish`` one second in (jobs 55663569/55663571,
#: 2026-08-30; the 72 MB it got through was still sitting in
#: ``rusted/scripts/`` afterwards). Named here instead, so a clone lands
#: where the batch's other data does rather than wherever a submission
#: happened to be typed.
#:
#: Per BATCH, not per project: two batches running at once would otherwise
#: hand the same ordinal to two different matches, which is the collision
#: this exists to remove rather than a smaller version of it.
CLONE_ROOT = "runs/clones"

#: The pinned game directory a workstation's clones are copied from.
#:
#: Here rather than beside each caller: the sweep entry point, the fleet
#: service's submitter and the clone check all name it, and three spellings of
#: one directory is two chances to look in the wrong place. Deliberately NOT
#: what a cluster member uses -- a compute node is told its staged tree.
PINNED_GAME_DIR = ".game"


def result_path(batch: str, job: SweepJob) -> str:
    """Return where one match files its scorecard, relative to a root.

    The one place this path is composed. A member's artifact and the command
    that writes it are both built from here, so the declaration and the run
    cannot disagree -- which is the failure
    :func:`~hpc3.contracts.job.require_artifact_in_command` exists to catch.

    Args:
        batch: The sweep this job belongs to.
        job: The job.

    Returns:
        The result path, relative and forward-slashed.
    """
    return f"{SWEEP_ROOT}/{batch}/{job_name(job)}{RESULT_SUFFIX}"


def trace_path(traces_root: str, batch: str, job: SweepJob) -> str:
    """Return where one match writes its per-sample trace.

    Args:
        traces_root: Where this run files traces. Relative to the repository
            on a workstation, absolute under the project's cluster directory
            on a node -- which is the whole reason it is an argument.
        batch: The sweep this job belongs to. Traces are namespaced by it
            because a job's name is only unique within one, and two sweeps
            sharing an arm label used to overwrite each other's records.
        job: The job.

    Returns:
        The trace path, forward-slashed.
    """
    return f"{traces_root}/{batch}/{job_name(job)}{TRACE_SUFFIX}"


def match_log_path(out_dir: str, job: SweepJob) -> str:
    """Return where one match's engine and agent logs go.

    Composed from the batch's own results directory rather than from the
    batch NAME, which is what it used to be. The two disagreed the moment
    they were not both relative to the repository: the runner created
    ``<out_dir>/logs`` with an absolute ``out_dir`` while the launcher was
    handed ``runs/sweeps/<batch>/logs``, so on a compute node the directory
    that existed and the directory written to were different ones.

    Args:
        out_dir: Where this batch's results are filed.
        job: The job.

    Returns:
        The log path, forward-slashed.
    """
    return f"{out_dir}/{LOG_DIR}/{job_name(job)}{LOG_SUFFIX}"


def cluster_path(root: str, project: str, relative: str) -> str:
    """Place a repository-relative path under a project's cluster directory.

    The boundary between the two roots, crossed in exactly one place. The
    project directory comes from :func:`hpc3.contracts.layout.project_dir`
    rather than being spelled here, so a batch's data lands beside the scripts
    and logs ``hpc3`` already puts there instead of in a parallel tree this
    package invented.

    Args:
        root: The cluster root, absolute and POSIX, from the hpc3 workspace.
        project: The hpc3 project this batch belongs to.
        relative: A repository-relative, forward-slashed path.

    Returns:
        The absolute cluster path.
    """
    return f"{project_dir(root, project)}/{relative}"


def clones_path(root: str, project: str, batch: str) -> str:
    """Return the directory one batch's game clones are made in.

    Args:
        root: The cluster root, absolute and POSIX, from the hpc3 workspace.
        project: The hpc3 project this batch belongs to.
        batch: The sweep whose members clone here.

    Returns:
        The absolute cluster path.
    """
    return cluster_path(root, project, f"{CLONE_ROOT}/{batch}")


def declares_result_for(declared: str, batch: str, job: SweepJob) -> bool:
    """Report whether a declared path is the one a given match writes.

    A suffix test rather than an equality, and deliberately: the running match
    is handed an absolute path and does not know the cluster root or the
    project that produced it, so the most it can check is that everything
    below them is its own. That still catches the failure worth catching -- a
    seed or an arm edited in the command and not in the artifact -- because
    both are in the part compared.

    Args:
        declared: The path the command line named.
        batch: The sweep this job belongs to.
        job: The job being played.

    Returns:
        True when the declared path ends with this match's own relative path.
    """
    return declared.endswith(result_path(batch, job))


__all__ = [
    "CLONE_ROOT",
    "GAME_DIR",
    "LOG_DIR",
    "LOG_SUFFIX",
    "PAYLOAD_DIR",
    "PINNED_GAME_DIR",
    "RESULT_SUFFIX",
    "SWEEP_ROOT",
    "TRACE_ROOT",
    "TRACE_SUFFIX",
    "clones_path",
    "cluster_path",
    "declares_result_for",
    "match_log_path",
    "result_path",
    "trace_path",
]
