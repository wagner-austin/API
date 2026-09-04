"""Bringing a finished run's verdict back from the node it ran on.

THIS FILE CLOSES A HOLE THE TESTS COULD NOT SEE. Before it existed,
:func:`fleet.core.dispatch.finish` and
:func:`fleet.core.dispatch.result_script` were written, covered and reachable
only from the test suite: NO command ever called them. So a dispatch could be
leased, staged, started and recorded, and then had no path at all from the
node's exit status back to the ledger -- every run stayed ``running`` until
its lease lapsed and ``fleet-watch`` reported it ``lost``. A hundred per cent
of statements and branches were exercised and the feature did not exist.

WHY COLLECTION IS A SEPARATE ACT AND NOT PART OF WATCHING. ``fleet-watch``
renders records; this mutates them. Folding the two would make the command a
session subscribes to a command that also closes leases, so a subscriber that
merely looked would be changing what the next capacity check sees. They stay
apart and compose in a shell loop, which is the same reason ``fleet-watch``
has no ``--follow``.

ABSENCE IS THE SIGNAL. The node writes its exit status only when the recipe
has returned, so a missing result file means the run is still going. There is
no heartbeat and nothing infers progress from elapsed time: a run that is
merely slow and a run that is wedged look identical from here, and the thing
that tells them apart is the lease expiring, which
:func:`fleet.cli.watch.lost_runs` already reports.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode
from typing_extensions import TypedDict

from fleet.contracts.ledger import LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.contracts.project import MAKE_TARGET, ProjectConfig, lease_seconds
from fleet.core import dispatch, remote

#: The exit status a passing ``make check`` leaves.
PASSING_EXIT_CODE = 0

#: What the result-reading script is called under a dispatch's directory.
#:
#: A distinct name from the build's own scripts so that reading a result
#: cannot overwrite the thing that produced it -- collection runs repeatedly
#: against a directory a build is still writing to.
POLL_SCRIPT_NAME = "collect.ps1"


class RunResult(TypedDict):
    """What a node reports about a dispatch that has finished.

    Attributes:
        exit_code: The recipe's exit status.
        finished_unix: When the build wrote it, whole seconds since the
            epoch, read from the node's own clock. Carried because the only
            question that matters about a finished run -- was its environment
            protected for the whole of it -- is answered against this and not
            against the moment somebody happened to collect.
    """

    exit_code: int
    finished_unix: int


def poll_result(node: NodeConfig, *, run_id: str) -> RunResult | None:
    """Ask a node how one dispatch ended, if it has.

    Args:
        node: The node it was dispatched to.
        run_id: The dispatch.

    Returns:
        Its status and finish time, or None while the run is still going.
        None is a real answer rather than an error: it is the ordinary state
        of every dispatch for as long as it runs.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            transport, or ``RUN_RESULT_UNREADABLE`` when the node answered
            with something that is not a status and a timestamp. The last is
            fatal rather than treated as unfinished: an unreadable result
            would otherwise make a finished run look like a running one
            forever, holding a node's budget against work that stopped.
    """
    target = f"{node['stage_root']}/{run_id}"
    answer = remote.run_script(
        node["host"],
        f"{target}/{POLL_SCRIPT_NAME}",
        dispatch.result_script(target),
    ).strip()
    if not answer:
        return None
    fields = answer.split()
    if len(fields) != 2 or not _is_exit_code(fields[0]) or not fields[1].isdigit():
        raise AppError(
            FleetErrorCode.RUN_RESULT_UNREADABLE,
            f"{run_id} on {node['host']} recorded {answer!r} where an exit status and a "
            f"timestamp were expected; the build's last act writes {dispatch.RESULT_NAME} and "
            "nothing else does, so this is a node that was written to by something other than "
            "the build",
        )
    return RunResult(exit_code=int(fields[0]), finished_unix=int(fields[1]))


def _is_exit_code(answer: str) -> bool:
    """Whether a node's answer is a whole number.

    Written rather than reached for via an exception, because the codebase
    does not use ``try``/``except`` to ask a question. A leading sign is
    accepted: PowerShell reports a process killed by an unhandled exception
    with a large negative status, and that is a result rather than a fault in
    the reading of it.

    Args:
        answer: What the node printed, already stripped.

    Returns:
        True when it is one optional sign followed by digits.
    """
    body = answer[1:] if answer[:1] in {"-", "+"} else answer
    return body.isdigit()


def outcome_for(exit_code: int) -> LedgerOutcome:
    """Say how a dispatch ended, from the status its recipe returned.

    Args:
        exit_code: The recipe's exit status.

    Returns:
        ``passed`` for zero and ``failed`` for anything else. A failing suite
        is a successful dispatch and the two are recorded separately for that
        reason: the ledger's ``outcome`` says what the WORK did, and the
        command that collected it exits 0 either way.
    """
    return "passed" if exit_code == PASSING_EXIT_CODE else "failed"


def describe(node: NodeConfig, *, run_id: str, exit_code: int) -> str:
    """Render what a closing ledger row and its feed event should say.

    Args:
        node: The node it ran on.
        run_id: The dispatch.
        exit_code: The recipe's exit status.

    Returns:
        One line carrying the status and the log's full remote path, so a
        subscriber reading only the feed can open the output without first
        working out which node ran it and where a stage directory goes.
    """
    return (
        f"make {MAKE_TARGET} exited {exit_code}; log at "
        f"{node['host']}:{node['stage_root']}/{run_id}/{dispatch.RESULT_NAME}.log"
    )


def lease_deadline(row: LedgerEntry, plan: ProjectConfig) -> int:
    """When the lease a dispatch took was due to expire.

    DERIVED FROM THE LEDGER ROW AND THE PLAN, not read from the lease file,
    and that is what makes it answerable at all. A lease is live state: it is
    removed on release and dropped on the next acquire, so by the time a
    result is collected the record of when it would have expired may be gone.
    The row's ``started_unix`` and the project's declared duration are the two
    inputs :func:`~fleet.core.dispatch.open_lease` used, so recomputing them
    reproduces the same instant without depending on the file still holding
    it.

    Args:
        row: The running row for the dispatch.
        plan: The project's declaration, whose expected duration sized the
            lease.

    Returns:
        The epoch second the claim was due to lapse.
    """
    return row["started_unix"] + lease_seconds(plan, slack=dispatch.LEASE_SLACK)


def outlived_its_lease(row: LedgerEntry, plan: ProjectConfig, *, finished_unix: int) -> bool:
    """Whether a run was still going after its claim had lapsed.

    THE ONLY DANGEROUS CASE, and the one worth refusing over: while the lease
    was gone and the build was still writing, a second dispatch could have
    been admitted into the same environment -- which is the corruption the
    whole package exists to prevent. Collecting LATE is not that, and treating
    the two alike refused a healthy run for having been read twenty minutes
    after it ended.

    Args:
        row: The running row for the dispatch.
        plan: The project's declaration.
        finished_unix: When the node says the build wrote its status.

    Returns:
        True when the build finished after the lease was due to lapse.
    """
    return finished_unix > lease_deadline(row, plan)


__all__ = [
    "PASSING_EXIT_CODE",
    "POLL_SCRIPT_NAME",
    "RunResult",
    "describe",
    "lease_deadline",
    "outcome_for",
    "outlived_its_lease",
    "poll_result",
]
