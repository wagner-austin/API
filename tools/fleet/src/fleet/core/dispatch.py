"""Starting a run on a node, and closing it out afterwards.

THE ORDER IS THE DESIGN. Lease, then stage, then launch, then record. Each
step is the precondition of the next and each has a failure that must not
leave the one before it half-done:

* The lease comes FIRST, before anything is copied. Staging into a project
  another dispatch holds would be pointless work at best, and at worst two
  extractions into one tree.
* The ledger row is written when the run STARTS, not when it ends, because a
  capacity check subtracts live rows from a node's free memory. A ledger that
  only recorded finished work would let the next dispatch onto a node the
  first had already filled.

WHY THE SUITE IS LAUNCHED THROUGH TASK SCHEDULER AND NOT AS AN SSH CHILD.
Windows OpenSSH assigns the session's process tree to a job object precisely
so the tree dies when the connection ends. A suite started as a child of the
dispatching ssh call therefore dies when that call returns -- which is
immediately, since the whole point is not to hold the connection for the
duration of a build. Measured on this fleet's own hardware and written up in
``memory/reference_long_runs_need_task_scheduler.md``: a ten-hour job launched
that way is leashed to a connection, and a process cannot be moved out of a
job object once it is in one.

The scheduler has its own trap, and ``-Priority 4`` is not decoration.
``Register-ScheduledTask`` defaults to priority 7, which sets below-normal CPU,
LOW I/O and background memory priority, all inherited by children; a run that
inherits it crawls, and the symptom looks like a slow node rather than a
misconfigured launch.

NOTHING HERE CATCHES. A failure to stage or launch propagates with its own
code, and :func:`finish` is the explicit act that gives the lease back -- at a
call site that knows the run is over, rather than a ``finally`` that would
also fire on the way out of a success.
"""

from __future__ import annotations

import pathlib

from fleet.contracts.feed import FeedEvent, FeedKind
from fleet.contracts.lease import Lease
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.contracts.project import MAKE_TARGET, ProjectConfig, lease_seconds
from fleet.core import _test_hooks, leases, records, remote, staging

#: How much longer than its estimate a dispatch may hold its lease.
#:
#: Two rather than a tighter figure because the estimate comes from whichever
#: machine last ran the suite, and this fleet's nodes differ by more than a
#: factor of two in free memory -- so a run on the smallest node legitimately
#: takes far longer than one on the largest. A lease that expired underneath a
#: healthy run would hand its environment to a second dispatch, which is the
#: corruption the lease exists to prevent, reintroduced by its own timeout.
LEASE_SLACK = 2.0

#: Where a dispatch's result is left on the node, under its own directory.
RESULT_NAME = "result.txt"


def run_id_for(project: str, *, started_unix: int) -> str:
    """Name a dispatch.

    Derived rather than random, so the identifier a person reads names the
    thing it identifies. The project's slashes become hyphens because the id
    is used as a directory name on the node.

    Args:
        project: Repo-relative project path.
        started_unix: When the dispatch began.

    Returns:
        The run id.
    """
    return f"{project.replace('/', '-')}-{started_unix}"


def launch_script(*, target: str, project: str, workers: int) -> str:
    """Render the script that starts the suite and does not wait for it.

    Args:
        target: Absolute remote directory holding the staged tree.
        project: Repo-relative project path.
        workers: Test workers the capacity check granted.

    Returns:
        The script's text. It registers a scheduled task, starts it, and
        returns -- so the ssh call that ran it can close without taking the
        build down with it.
    """
    working = f"{target}/{project}"
    inner = (
        f"cd '{working}'; "
        f"$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '{workers}'; "
        f"make {MAKE_TARGET} *> '{target}/{RESULT_NAME}.log'; "
        f"$LASTEXITCODE | Set-Content -LiteralPath '{target}/{RESULT_NAME}'"
    )
    return (
        f"$action = New-ScheduledTaskAction -Execute 'powershell.exe' "
        f"-Argument '-NoProfile -ExecutionPolicy Bypass -Command \"{inner}\"'\n"
        f"$settings = New-ScheduledTaskSettingsSet -Priority 4 "
        f"-ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew\n"
        f"$principal = New-ScheduledTaskPrincipal "
        f"-UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) "
        f"-LogonType S4U\n"
        f"Register-ScheduledTask -TaskName 'fleet-{_task_name(target)}' -Action $action "
        f"-Settings $settings -Principal $principal -Force | Out-Null\n"
        f"Start-ScheduledTask -TaskName 'fleet-{_task_name(target)}'\n"
        f"Write-Output 'launched'\n"
    )


def _task_name(target: str) -> str:
    """Name the scheduled task after the dispatch that owns it.

    Args:
        target: Absolute remote directory holding the staged tree.

    Returns:
        The task's name, which is the directory's last segment.
    """
    return target.rsplit("/", 1)[-1]


def result_script(target: str) -> str:
    """Render the script that reports whether a dispatch has finished.

    Args:
        target: Absolute remote directory holding the staged tree.

    Returns:
        The script's text. It prints the recorded exit status, or nothing at
        all while the run is still going -- absence is the signal, so a run
        that has not written its status yet cannot be mistaken for one that
        exited zero.
    """
    return (
        f"if (Test-Path -LiteralPath '{target}/{RESULT_NAME}') "
        f"{{ Get-Content -Raw -LiteralPath '{target}/{RESULT_NAME}' }}\n"
    )


def open_lease(
    *,
    node: str,
    project: str,
    run_id: str,
    agent: str,
    session_id: str,
    plan: ProjectConfig,
    now_unix: int,
) -> Lease:
    """Build the claim a dispatch will hold for its run.

    Args:
        node: The node's workspace name.
        project: Repo-relative project path.
        run_id: The dispatch.
        agent: Board label of the dispatching session.
        session_id: That session's UUID.
        plan: The project, whose expected duration sizes the window.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        The lease, sized at :data:`LEASE_SLACK` times the estimate.
    """
    return Lease(
        node=node,
        project=project,
        run_id=run_id,
        agent=agent,
        session_id=session_id,
        acquired_unix=now_unix,
        expires_unix=now_unix + lease_seconds(plan, slack=LEASE_SLACK),
    )


def started_row(
    *,
    lease: Lease,
    host: str,
    workers: int,
    detail: str,
) -> LedgerEntry:
    """Build the ledger row a dispatch is recorded as running by.

    Args:
        lease: The claim it holds, which carries its identity.
        host: The SSH alias actually used.
        workers: Test workers granted.
        detail: What to say about it.

    Returns:
        The row, whose ``ended_unix`` equals its start because it has not
        ended. A nullable timestamp would make every reader branch, and the
        outcome field already says whether the run is over.
    """
    return LedgerEntry(
        run_id=lease["run_id"],
        node=lease["node"],
        host=host,
        project=lease["project"],
        agent=lease["agent"],
        session_id=lease["session_id"],
        started_unix=lease["acquired_unix"],
        ended_unix=lease["acquired_unix"],
        outcome="running",
        exit_code=NO_EXIT_CODE,
        workers=workers,
        detail=detail,
    )


def closed_row(
    row: LedgerEntry,
    *,
    outcome: LedgerOutcome,
    exit_code: int,
    ended_unix: int,
    detail: str,
) -> LedgerEntry:
    """Build the row that supersedes a running one.

    Appended rather than replacing, because the ledger is append-only: the
    running row is history, and a reader wanting the current state takes the
    last row for that id.

    Args:
        row: The running row.
        outcome: How it ended.
        exit_code: The recipe's status.
        ended_unix: When it ended.
        detail: What to say about it.

    Returns:
        The closing row.
    """
    return LedgerEntry(
        run_id=row["run_id"],
        node=row["node"],
        host=row["host"],
        project=row["project"],
        agent=row["agent"],
        session_id=row["session_id"],
        started_unix=row["started_unix"],
        ended_unix=ended_unix,
        outcome=outcome,
        exit_code=exit_code,
        workers=row["workers"],
        detail=detail,
    )


def emit(
    feed_path: pathlib.Path,
    *,
    lease: Lease,
    kind: FeedKind,
    detail: str,
    now_unix: int,
) -> None:
    """Append one event to the stream subscribers tail.

    Args:
        feed_path: The feed file.
        lease: The dispatch the event belongs to.
        kind: What happened. Typed as the Literal rather than a string, so a
            kind that does not exist is a type error here rather than a
            decode failure in whoever reads the feed next.
        detail: Human-readable specifics.
        now_unix: Current time, whole seconds since the epoch.
    """
    records.append_feed(
        feed_path,
        FeedEvent(
            at_unix=now_unix,
            run_id=lease["run_id"],
            node=lease["node"],
            project=lease["project"],
            kind=kind,
            detail=detail,
        ),
    )


def start(
    loaded_leases: pathlib.Path,
    loaded_ledger: pathlib.Path,
    loaded_feed: pathlib.Path,
    *,
    node_name: str,
    node: NodeConfig,
    project: str,
    plan: ProjectConfig,
    workers: int,
    agent: str,
    session_id: str,
    project_root: pathlib.Path,
    archive_dir: pathlib.Path,
) -> LedgerEntry:
    """Take the lease, stage the tree, launch the suite, and record it.

    Args:
        loaded_leases: The lease file.
        loaded_ledger: The ledger file.
        loaded_feed: The feed file.
        node_name: The node's workspace name.
        node: Its declaration.
        project: Repo-relative project path.
        plan: The project's declaration.
        workers: Test workers the capacity check granted.
        agent: Board label of the dispatching session.
        session_id: That session's UUID.
        project_root: Absolute path to the monorepo root.
        archive_dir: Local directory to build the archive in. The file is
            named after the run, so two concurrent dispatches cannot write
            one archive over each other.

    Returns:
        The running ledger row.

    Raises:
        AppError: With ``LEASE_HELD`` when another dispatch holds this
            project on this node, ``STAGE_ARCHIVE_UNREADABLE`` or
            ``STAGE_DIGEST_MISMATCH`` from staging, or ``NODE_UNREACHABLE``
            or ``DISPATCH_FAILED`` from the transport.
    """
    now_unix = _test_hooks.now()
    run_id = run_id_for(project, started_unix=now_unix)
    lease = open_lease(
        node=node_name,
        project=project,
        run_id=run_id,
        agent=agent,
        session_id=session_id,
        plan=plan,
        now_unix=now_unix,
    )
    leases.acquire(loaded_leases, lease, now_unix=now_unix)
    emit(loaded_feed, lease=lease, kind="leased", detail=f"{workers} worker(s)", now_unix=now_unix)

    payload = staging.archive(project_root, project, archive_dir / f"{run_id}.tgz")
    target = staging.stage(
        node["host"], run_id=run_id, stage_root=node["stage_root"], payload=payload
    )
    emit(
        loaded_feed,
        lease=lease,
        kind="staged",
        detail=f"{len(payload)} bytes to {target}",
        now_unix=_test_hooks.now(),
    )

    remote.run_script(
        node["host"],
        f"{target}/launch.ps1",
        launch_script(target=target, project=project, workers=workers),
    )
    row = started_row(lease=lease, host=node["host"], workers=workers, detail=f"staged to {target}")
    records.append_ledger(loaded_ledger, row)
    emit(
        loaded_feed,
        lease=lease,
        kind="started",
        detail=f"make {MAKE_TARGET} at {workers} worker(s)",
        now_unix=_test_hooks.now(),
    )
    return row


def finish(
    loaded_leases: pathlib.Path,
    loaded_ledger: pathlib.Path,
    loaded_feed: pathlib.Path,
    *,
    row: LedgerEntry,
    lease: Lease,
    outcome: LedgerOutcome,
    exit_code: int,
    detail: str,
) -> LedgerEntry:
    """Close a dispatch out: record the outcome, emit it, release the lease.

    THE LEASE IS RELEASED LAST. If the ledger write failed after the lease had
    already gone, the environment would be free while the record still said
    ``running``, and the next capacity check would count a dispatch that no
    longer exists. In this order a failure leaves the lease held, which
    expires on its own and is the safe direction to fail in.

    Args:
        loaded_leases: The lease file.
        loaded_ledger: The ledger file.
        loaded_feed: The feed file.
        row: The running row this supersedes.
        lease: The claim to give back.
        outcome: How the dispatch ended.
        exit_code: The recipe's status, or
            :data:`~fleet.contracts.ledger.NO_EXIT_CODE` when there was none.
        detail: What to say about it.

    Returns:
        The closing row.

    Raises:
        AppError: With ``LEASE_NOT_HELD`` if no live lease names this run,
            which means it expired while the run was still going and another
            dispatch may already be inside the same environment.
    """
    ended_unix = _test_hooks.now()
    closing = closed_row(
        row, outcome=outcome, exit_code=exit_code, ended_unix=ended_unix, detail=detail
    )
    records.append_ledger(loaded_ledger, closing)
    emit(loaded_feed, lease=lease, kind=_OUTCOME_EVENT[outcome], detail=detail, now_unix=ended_unix)
    leases.release(loaded_leases, run_id=row["run_id"], now_unix=ended_unix)
    return closing


#: Which feed kind announces each terminal outcome.
#:
#: A mapping rather than a name shared between the two vocabularies, because
#: they are not the same vocabulary: ``running`` is a ledger outcome with no
#: event, and ``leased`` / ``staged`` / ``started`` are events with no outcome.
#: Spelling the overlap once, here, is what stops a reader assuming the two
#: enums are interchangeable.
_OUTCOME_EVENT: dict[LedgerOutcome, FeedKind] = {
    "refused": "refused",
    "passed": "passed",
    "failed": "failed",
    "cancelled": "cancelled",
    "lost": "lost",
    "running": "heartbeat",
}


__all__ = [
    "LEASE_SLACK",
    "RESULT_NAME",
    "closed_row",
    "emit",
    "finish",
    "launch_script",
    "open_lease",
    "result_script",
    "run_id_for",
    "start",
    "started_row",
]
