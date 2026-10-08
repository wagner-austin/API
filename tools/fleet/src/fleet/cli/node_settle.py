"""Settle one ended run: its verdict, its queue close, its ledger row, its directory.

Split out of :mod:`fleet.cli.node_collect`, which had reached the file
ceiling, when its order became the point (MCPs board task 8993c306).

THE QUEUE IS CLOSED BEFORE ANYTHING ON THIS MACHINE OR THE NODE CHANGES.
Until 2026-10-05 the settle retired the run's directory, finished its ledger
row and only then closed its queue job. A queue that did not answer that
close (a deploy recreating mcp-fleet, :mod:`fleet.core.queue_transport`) left
the job running on the queue with no live row behind it, which every later
collect pass leaves alone, so the waiting session waited out the claim's
hour-long lease; and a retire done before a refused call cannot be done
again, since the result it reads goes with the directory. Now every queue
and board call comes first: a refusal anywhere in them leaves the row live
and the directory where it was, so the next poll settles the run again from
the start, as if it had just ended. The verdict names the transcript where
the retire will keep it (:func:`fleet.core.names.retained_log_path`, the
path :func:`fleet.core.retire.retire_on_node` returns).

What a refusal can still repeat: a task-less job's verdict, posted to its
submitter's feed before the close, is posted again when the close is refused
and the settle runs again. A second line is visible and harmless; a verdict
posted only after a close that then failed would be one nobody ever saw.

After the close, the ledger row is finished (this machine's files) and the
directory retired (the node). A RETIRE THE NODE DID NOT ANSWER IS OWED, NOT
RAISED (MCPs board task 8776b828): the row and the queue job are already
closed, so no later settle would retry it, and raising ended the serve and
every run it held. It is recorded beside the ledger and each later collect
pass sends it until the node answers (:mod:`fleet.core.retire_owed`); until
then the verdict names a transcript not yet moved. A retire the node
answered and failed still raises, so the serve's log says so.

THE TAIL IS READ FIRST, AS A READ THE NODE MAY MISS (MCPs board task
c1d48330). The settle read the transcript's tail with a raising read, and
on 2026-10-07 at 02:54:10Z lavender-wsl stopped answering ssh as row
1d5d0a1a's run ended: the collect pass's tail read raised NODE_UNREACHABLE,
the serve ended with it, and the row closed at the next start, 192 s after
its check. It is now read first, with :func:`read_tail`, a read the node
may miss, and a node that did not answer changes nothing and leaves the run
to be read again at the next poll, as the result read already did
(:func:`fleet.core.collect.attempt_poll_result`).
"""

from __future__ import annotations

from platform_core.error_codes_fleet import FleetErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config
from fleet.cli.node_collected import Collected, CollectOutcome
from fleet.contracts.dispatch import ClosingStatus, DispatchJob, encode_job_line
from fleet.contracts.ledger import LedgerEntry
from fleet.contracts.node import NodeConfig
from fleet.core import (
    collect,
    dialect,
    dispatch,
    names,
    queue,
    remote,
    retire_owed,
    venv_sweep,
    verdict,
)


def settle(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    row: LedgerEntry,
    node: NodeConfig,
    exit_code: int,
    ended_unix: int,
    detail: str,
    stopped: str | None,
) -> Collected:
    """Read a run's tail and judge it, post a task-less run's verdict, close its
    queue job, then finish its row, retire its directory and sweep the node's
    orphaned virtualenvs.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers, for the verdict.
        job: The queue job this runner holds.
        identity: This runner's identity arguments.
        row: The run's live ledger row.
        node: The node it ran on.
        exit_code: The status to record.
        ended_unix: When the check ended: its result's time, or the stop's.
        detail: What the ledger and the feed say about it.
        stopped: Why the runner stopped the build, appended to the verdict
            line, or None when the build finished on its own. A unit-end
            line in the tail (:func:`fleet.core.verdict.read_unit_end`) is
            appended before it.

    Returns:
        :attr:`~fleet.cli.node_collected.CollectOutcome.SETTLED` with the
        job's line and the verdict, as posted and as the queue job's detail;
        or :attr:`~fleet.cli.node_collected.CollectOutcome.UNREACHABLE`, with
        nothing anywhere changed, when the node did not answer the tail's read.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the job carries no sha;
            ``QUEUE_UNANSWERED`` when the board or the queue did not answer,
            with nothing on this machine or the node changed;
            ``DISPATCH_FAILED`` when the node answered the retire and it
            failed there, with the row and the job already closed; or a
            node, board or queue failure. Not caught. A node that did not
            answer the retire raises nothing: the retire is owed.
    """
    sha = require_sha(job)
    read = read_tail(node, run_id=row["run_id"])
    tail = read["output"]
    if tail is None:
        return Collected(
            outcome=CollectOutcome.UNREACHABLE,
            line=(
                f"{encode_job_line(job)}: did not answer the read of its transcript's tail: "
                f"{read['unreachable']}"
            ),
        )
    judged = verdict.judge(
        job_id=job["job_id"],
        project=row["project"],
        sha=sha,
        node=row["node"],
        exit_code=exit_code,
        tail=tail,
        ended_unix=ended_unix,
        log_path=names.retained_log_path(node["stage_root"], row["run_id"]),
        run_id=row["run_id"],
    )
    rendered = verdict.render_verdict(judged)
    # How the unit or task ended, when the build did not write its own
    # status: an oom-kill or a killed powershell.exe reads as one rather than
    # as a failing suite (MCPs board tasks c8585623 and 4e3afe4f,
    # fleet.core.linux_unit_end and fleet.core.windows_result).
    unit_end = verdict.read_unit_end(tail)
    ended = rendered if unit_end is None else f"{rendered} {unit_end}"
    line = ended if stopped is None else f"{ended} stopped: {stopped}"
    # A job naming a task gets its row from the queue's close below, on the
    # task's thread and addressed to the submitter (MCPs board task
    # 2fecad69); only a task-less job's verdict is this runner's to post.
    if job["task_id"] is None:
        queue.post_verdict(
            board,
            submitted_by=job["submitted_by"],
            line=line,
            identity=identity,
        )
    queue.report_close(
        credentials,
        job_id=job["job_id"],
        status=ClosingStatus.PASSED if exit_code == 0 else ClosingStatus.FAILED,
        exit_code=exit_code,
        detail=line,
        identity=identity,
    )
    dispatch.finish(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        row=row,
        outcome=collect.outcome_for(exit_code),
        exit_code=exit_code,
        detail=detail,
    )
    # After the queue close, so the session waiting on this row is not kept
    # waiting on housekeeping (fleet.core.retire, MCPs board task 8993c306).
    # A node that did not answer owes the retire to a later collect pass,
    # which sweeps after it (MCPs board task 8776b828).
    if retire_owed.retire_or_owe(loaded.retires, node, alias=row["node"], run_id=row["run_id"]):
        venv_sweep.sweep_on_node(node)
    return Collected(outcome=CollectOutcome.SETTLED, line=f"{encode_job_line(job)}: {line}")


def read_tail(node: NodeConfig, *, run_id: str) -> remote.Answered:
    """Read the tail of a run's transcript, the lines its verdict is judged from.

    Args:
        node: The node it ran on.
        run_id: The run.

    Returns:
        The last :data:`fleet.core.verdict.LOG_TAIL_LINES` lines, or why the
        node did not answer.

    Raises:
        AppError: ``DISPATCH_FAILED`` when the node answered and the read
            failed there.
    """
    target = names.dispatch_directory(node["stage_root"], run_id)
    spoken = dialect.for_platform(node["platform"])
    return remote.read_script(
        node["host"],
        spoken.script_path(target, names.LOG_TAIL_STEM),
        spoken.log_tail_script(target, verdict.LOG_TAIL_LINES),
        platform=node["platform"],
    )


def require_sha(job: DispatchJob) -> str:
    """The commit a node-lane job names.

    Args:
        job: The claimed job.

    Returns:
        Its sha.

    Raises:
        AppError: ``QUEUE_ANSWER_MALFORMED`` when the job carries none,
            which the queue's pin (MCPs mig 532) makes impossible for a
            node-lane row; a null here means the contract moved.
    """
    sha = job["sha"]
    if sha is None:
        raise AppError(
            code=FleetErrorCode.QUEUE_ANSWER_MALFORMED,
            message=f"node-lane job {job['job_id']} carries no sha; the queue's pin forbids it",
        )
    return sha


__all__ = ["read_tail", "require_sha", "settle"]
