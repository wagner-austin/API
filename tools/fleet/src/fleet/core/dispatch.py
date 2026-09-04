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
from fleet.core import _test_hooks, leases, manifest, records, remote, staging

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

#: What the build's own script is called under a dispatch's directory.
#:
#: THE BUILD IS A SEPARATE FILE FROM THE THING THAT SCHEDULES IT, and that is
#: the fix for a real bug rather than a tidiness preference. The first version
#: passed the whole build as ``-Argument '-Command "cd ''{path}''; ..."'``, and
#: PowerShell ended the single-quoted argument at the first inner quote: the
#: registered task carried ``-Command "cd`` as its arguments and the remaining
#: two hundred characters as its WORKING DIRECTORY. Measured on sedona
#: 2026-09-04, the resulting task could not be started at all -- ``Element not
#: found``, because that working directory does not exist.
#:
#: Sending the script and naming it by path is the same rule
#: :mod:`fleet.core.remote` follows for ssh, applied one layer further in. The
#: registration then interpolates one path and no code.
BUILD_SCRIPT_NAME = "build.ps1"

#: What the script that registers and starts the task is called.
REGISTER_SCRIPT_NAME = "register.ps1"

#: Task Scheduler's ``SCHED_S_TASK_HAS_NOT_RUN``, 0x00041303.
#:
#: The status a registered task reports until it has run once. It is the
#: signal :func:`register_script` waits to stop seeing, because
#: ``Start-ScheduledTask`` reports a failure to start as a NON-TERMINATING
#: error: PowerShell prints it, exits 0, and the dispatch records a run that
#: does not exist. Measured 2026-09-04 -- the ledger said ``running`` for a
#: task whose ``LastRunTime`` was still the 1999 sentinel.
TASK_HAS_NOT_RUN = 267011

#: How long the node waits for a started task to leave that state.
#:
#: Generous because it is bounding a Task Scheduler round trip and not any
#: work: the task only has to BEGIN. A build that starts and fails in the
#: first second still leaves this state, so the wait ends on the first status
#: change rather than on success.
LAUNCH_TIMEOUT_SECONDS = 30


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


def build_script(*, target: str, project: str, workers: int) -> str:
    """Render the script that runs the suite, which is all it does.

    It knows nothing about scheduling. That separation is what keeps the
    registration free of nested quoting -- see :data:`BUILD_SCRIPT_NAME`.

    ``$LASTEXITCODE`` rather than ``$?`` because the recipe is a native
    program: PowerShell sets ``$?`` false whenever a native command writes to
    a redirected stderr, which ``make`` does routinely on a passing run.

    Args:
        target: Absolute remote directory holding the staged tree.
        project: Repo-relative project path.
        workers: Test workers the capacity check granted.

    Returns:
        The script's text. Its last act writes the recipe's exit status to
        :data:`RESULT_NAME`, which is what
        :func:`~fleet.core.collect.poll_result` later reads -- so the file's
        ABSENCE means the run is still going and its presence means it is
        over, with no third state to disambiguate.
    """
    return (
        f"$ErrorActionPreference = 'Continue'\n"
        f"Set-Location -LiteralPath '{target}/{project}'\n"
        f"$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '{workers}'\n"
        f"make {MAKE_TARGET} *> '{target}/{RESULT_NAME}.log'\n"
        f"$LASTEXITCODE | Set-Content -LiteralPath '{target}/{RESULT_NAME}'\n"
    )


def register_script(*, target: str, run_id: str) -> str:
    """Render the script that schedules the build and proves it started.

    ``-AllowStartIfOnBatteries`` and ``-DontStopIfGoingOnBatteries`` are not
    optional here and their defaults are the wrong way round for this fleet.
    ``New-ScheduledTaskSettingsSet`` defaults both battery settings to
    refusing, and two of the three nodes are laptops -- so a dispatch to an
    unplugged sedona would register a task that never runs, or would have a
    running suite killed the moment somebody unplugged it, in both cases
    reporting nothing.

    Args:
        target: Absolute remote directory holding the staged tree.
        run_id: The dispatch, which names its own task.

    Returns:
        The script's text. It registers the task, starts it, and then WAITS
        for the task to leave :data:`TASK_HAS_NOT_RUN` before saying so --
        because ``Start-ScheduledTask`` reports a refusal as a non-terminating
        error that would otherwise exit 0 and be recorded as a launch.
    """
    task = task_name(run_id)
    return (
        f"$ErrorActionPreference = 'Stop'\n"
        f"$action = New-ScheduledTaskAction -Execute 'powershell.exe' "
        f"-Argument '-NoProfile -ExecutionPolicy Bypass -File \"{target}/{BUILD_SCRIPT_NAME}\"'\n"
        f"$settings = New-ScheduledTaskSettingsSet -Priority 4 "
        f"-ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew "
        f"-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries\n"
        f"$principal = New-ScheduledTaskPrincipal "
        f"-UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) "
        f"-LogonType S4U\n"
        f"Register-ScheduledTask -TaskName '{task}' -Action $action "
        f"-Settings $settings -Principal $principal -Force | Out-Null\n"
        f"Start-ScheduledTask -TaskName '{task}'\n"
        f"$deadline = (Get-Date).AddSeconds({LAUNCH_TIMEOUT_SECONDS})\n"
        f"while ((Get-Date) -lt $deadline) {{\n"
        f"  if ((Get-ScheduledTaskInfo -TaskName '{task}').LastTaskResult "
        f"-ne {TASK_HAS_NOT_RUN}) {{ Write-Output 'launched'; exit 0 }}\n"
        f"  Start-Sleep -Milliseconds 500\n"
        f"}}\n"
        f'throw "{task} registered but has not run after {LAUNCH_TIMEOUT_SECONDS}s"\n'
    )


def task_name(run_id: str) -> str:
    """Name the scheduled task a dispatch owns.

    Derived from the run id rather than from the stage path, so the name a
    dispatch registers and the name :mod:`fleet.cli.cancel` stops are the same
    string produced by the same function. They were separately spelled before,
    which is a rename away from a cancel that silently stops nothing.

    Args:
        run_id: The dispatch.

    Returns:
        The task's name.
    """
    return f"fleet-{run_id}"


def result_script(target: str) -> str:
    """Render the script that reports whether a dispatch has finished.

    IT REPORTS *WHEN* AS WELL AS *WHAT*, and the timestamp is not decoration.
    Whether a run was safe is a question about whether its lease covered the
    whole of it, and that can only be answered against the moment the build
    ended -- which the node knows and nobody else does. Asking only for the
    status forces the reader to substitute "is a lease held right now", which
    is a question about how promptly somebody collected: measured 2026-09-04,
    a run that finished three minutes inside its window was refused twenty
    minutes later for having been collected late.

    Args:
        target: Absolute remote directory holding the staged tree.

    Returns:
        The script's text. It prints the exit status and the epoch second the
        result was written, space-separated -- or nothing at all while the run
        is still going. Absence is the signal, so a run that has not written
        its status cannot be mistaken for one that exited zero.

        The epoch is computed by subtracting the Unix epoch from a UTC
        timestamp rather than with ``-UFormat %s``, which in PowerShell 5.1
        converts from LOCAL time and would put every node's answer out by its
        own offset.
    """
    return (
        f"if (Test-Path -LiteralPath '{target}/{RESULT_NAME}') {{\n"
        f"  $file = Get-Item -LiteralPath '{target}/{RESULT_NAME}'\n"
        f"  $code = (Get-Content -Raw -LiteralPath '{target}/{RESULT_NAME}').Trim()\n"
        f"  $epoch = [int]($file.LastWriteTimeUtc - [datetime]'1970-01-01').TotalSeconds\n"
        f'  "$code $epoch"\n'
        f"}}\n"
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
    run_id: str,
    node: str,
    project: str,
    kind: FeedKind,
    detail: str,
    now_unix: int,
) -> None:
    """Append one event to the stream subscribers tail.

    It takes the three identifying strings rather than a
    :class:`~fleet.contracts.lease.Lease`, because a lease is not what an
    event is about: the terminal events are emitted when a run is CLOSED, at
    which point the caller may hold a ledger row and no lease at all. Naming
    the fields is what lets both callers use the one function.

    Args:
        feed_path: The feed file.
        run_id: The dispatch the event belongs to.
        node: Its node's workspace name.
        project: Repo-relative project path.
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
            run_id=run_id,
            node=node,
            project=project,
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
    emit(
        loaded_feed,
        run_id=run_id,
        node=node_name,
        project=project,
        kind="leased",
        detail=f"{workers} worker(s)",
        now_unix=now_unix,
    )

    members = manifest.build_tree(project_root, project)
    payload = staging.archive(project_root, members, archive_dir / f"{run_id}.tgz")
    target = staging.stage(
        node["host"], run_id=run_id, stage_root=node["stage_root"], payload=payload
    )
    emit(
        loaded_feed,
        run_id=run_id,
        node=node_name,
        project=project,
        kind="staged",
        # Counted, not listed. A real dispatch carries about forty-six members
        # -- the project, its dependencies, and one manifest per package for
        # the guard rules that scan them all -- and naming each turned one feed
        # line into two thousand characters. The members are derivable from
        # `fleet.core.manifest`; the feed's job is to say a run reached this
        # step, with enough to spot a stage that carried the wrong amount.
        detail=f"{len(payload)} bytes in {len(members)} member(s) to {target}",
        now_unix=_test_hooks.now(),
    )

    remote.send_script(
        node["host"],
        f"{target}/{BUILD_SCRIPT_NAME}",
        build_script(target=target, project=project, workers=workers),
    )
    remote.run_script(
        node["host"],
        f"{target}/{REGISTER_SCRIPT_NAME}",
        register_script(target=target, run_id=run_id),
    )
    row = started_row(lease=lease, host=node["host"], workers=workers, detail=f"staged to {target}")
    records.append_ledger(loaded_ledger, row)
    emit(
        loaded_feed,
        run_id=run_id,
        node=node_name,
        project=project,
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

    A LEASE THAT HAS ALREADY LAPSED IS NOT AN ERROR HERE, and requiring one
    was a defect. Whether a run was PROTECTED is a question about whether its
    lease covered the run; whether a lease is held NOW is a question about how
    promptly somebody came to collect. Conflating them refused a run that had
    finished three minutes inside its window because it was collected twenty
    minutes later. The genuine hazard -- a run that outlived its lease while
    still going -- is decided by the caller against the moment the build
    ended, before it gets here.

    Args:
        loaded_leases: The lease file.
        loaded_ledger: The ledger file.
        loaded_feed: The feed file.
        row: The running row this supersedes.
        outcome: How the dispatch ended.
        exit_code: The recipe's status, or
            :data:`~fleet.contracts.ledger.NO_EXIT_CODE` when there was none.
        detail: What to say about it.

    Returns:
        The closing row.
    """
    ended_unix = _test_hooks.now()
    closing = closed_row(
        row, outcome=outcome, exit_code=exit_code, ended_unix=ended_unix, detail=detail
    )
    records.append_ledger(loaded_ledger, closing)
    emit(
        loaded_feed,
        run_id=row["run_id"],
        node=row["node"],
        project=row["project"],
        kind=_OUTCOME_EVENT[outcome],
        detail=detail,
        now_unix=ended_unix,
    )
    leases.release_if_held(loaded_leases, run_id=row["run_id"], now_unix=ended_unix)
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
    "BUILD_SCRIPT_NAME",
    "LAUNCH_TIMEOUT_SECONDS",
    "LEASE_SLACK",
    "REGISTER_SCRIPT_NAME",
    "RESULT_NAME",
    "TASK_HAS_NOT_RUN",
    "build_script",
    "closed_row",
    "emit",
    "finish",
    "open_lease",
    "register_script",
    "result_script",
    "run_id_for",
    "start",
    "started_row",
    "task_name",
]
