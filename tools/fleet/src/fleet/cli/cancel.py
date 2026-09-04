"""CLI: stop a dispatch, and give its project's environment back.

Usage:
    fleet-cancel --config fleet.json --run services-Model-Trainer-1757000000

WHAT IT ACTUALLY DOES, in the order that matters. It stops the scheduled task
on the node, closes the ledger row as ``cancelled``, emits that on the feed,
and releases the lease last -- so a failure part-way leaves the lease HELD,
which expires on its own. Releasing first and failing after would free the
environment while the record still said ``running``, and the next capacity
check would count a dispatch that no longer exists.

IT IS THE ONLY COMMAND THAT KILLS ANYTHING, and it kills exactly one dispatch
by name. There is no sweep, no "cancel everything on this node", and no age
heuristic. That is a direct response to what this fleet is for: the incident
behind the package involved processes being destroyed by something that was
not trying to destroy them, and a tool that could take out work it did not
start would be the same hazard wearing a badge.

A DISPATCH NOBODY IS HOLDING IS STILL CANCELLABLE. A run whose lease expired
while it was still going is precisely the wedge case, and refusing to cancel
it because the lease is gone would leave the only tool that can stop it
unable to. So the lease release is allowed to find nothing, and says so.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.errors import AppError, FleetErrorCode
from platform_core.logging import get_logger, setup_logging

from fleet.cli import _config
from fleet.contracts.feed import FeedEvent
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, is_live
from fleet.contracts.workspace import require_node
from fleet.core import _test_hooks, dispatch, leases, records, remote

_log = get_logger(__name__)

RUN_FLAG = "--run"

_FLAGS = (_config.CONFIG_FLAG, RUN_FLAG)


def find_live_row(loaded: _config.LoadedWorkspace, *, run_id: str) -> LedgerEntry:
    """Find the running row for one dispatch.

    The LAST matching row wins, because the ledger is append-only and a
    closing row supersedes its running one. Taking the first would find a
    dispatch that has already finished and try to cancel it again.

    Args:
        loaded: The workspace and its resolved record paths.
        run_id: The dispatch.

    Returns:
        Its most recent row.

    Raises:
        AppError: With ``RUN_UNKNOWN`` when the ledger holds no row for that
            id, or holds one that has already ended. Two conditions, one
            code, because the caller's next act is the same either way: look
            at ``fleet-watch`` to see what happened to it.
    """
    matching = [row for row in records.read_ledger(loaded.ledger) if row["run_id"] == run_id]
    if not matching:
        raise AppError(
            FleetErrorCode.RUN_UNKNOWN,
            f"no dispatch named {run_id!r} in {loaded.ledger}",
        )
    latest = matching[-1]
    if not is_live(latest):
        raise AppError(
            FleetErrorCode.RUN_UNKNOWN,
            f"{run_id} already ended as {latest['outcome']!r}; nothing to cancel",
        )
    return latest


def stop_script(run_id: str) -> str:
    """Render the script that stops one dispatch's task on the node.

    ``-Confirm:$false`` because there is nobody at the node to answer, and an
    unanswered prompt would hang this command until its ssh timeout rather
    than stopping anything.

    The task's name comes from :func:`fleet.core.dispatch.task_name`, the
    same function the dispatch registered it with. Spelling it here a second
    time is one rename away from a cancel that reports success having stopped
    nothing.

    Args:
        run_id: The dispatch, which names its own task.

    Returns:
        The script's text.
    """
    task = dispatch.task_name(run_id)
    return (
        f"Stop-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue\n"
        f"Unregister-ScheduledTask -TaskName '{task}' -Confirm:$false "
        f"-ErrorAction SilentlyContinue\n"
        f"Write-Output 'stopped {task}'\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Cancel one dispatch.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the task is stopped and the row is closed.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing its value.
        AppError: With ``RUN_UNKNOWN`` if no live dispatch has that id,
            ``WORKSPACE_NODE_UNKNOWN`` if its node has since left the
            workspace, or a transport code if the node cannot be reached.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)
    run_id = cli_args.require_flag(parsed, RUN_FLAG)

    row = find_live_row(loaded, run_id=run_id)
    node = require_node(loaded.workspace, row["node"])
    remote.run_script(
        node["host"],
        f"{node['stage_root']}/stop-{run_id}.ps1",
        stop_script(run_id),
    )

    ended_unix = _test_hooks.now()
    records.append_ledger(
        loaded.ledger,
        dispatch.closed_row(
            row,
            outcome="cancelled",
            exit_code=NO_EXIT_CODE,
            ended_unix=ended_unix,
            detail="cancelled by fleet-cancel",
        ),
    )
    _emit_cancelled(loaded, row=row, ended_unix=ended_unix)
    released = leases.release_if_held(loaded.leases, run_id=run_id, now_unix=ended_unix)

    _log.info("cancelled %s on %s (%s)", run_id, row["node"], released)
    return 0


def _emit_cancelled(loaded: _config.LoadedWorkspace, *, row: LedgerEntry, ended_unix: int) -> None:
    """Announce a cancellation on the feed.

    Args:
        loaded: The workspace and its resolved record paths.
        row: The row being closed.
        ended_unix: When it was cancelled.
    """
    records.append_feed(
        loaded.feed,
        FeedEvent(
            at_unix=ended_unix,
            run_id=row["run_id"],
            node=row["node"],
            project=row["project"],
            kind="cancelled",
            detail=f"cancelled by fleet-cancel; was dispatched by {row['agent']}",
        ),
    )


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="fleet-cancel",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["entrypoint", "find_live_row", "main", "stop_script"]


# Without this, `python -m fleet.cli.cancel` imports the module, runs nothing
# and exits 0 -- which reads as a cancellation that never happened, leaving a
# suite running that somebody believes they stopped.
if __name__ == "__main__":
    entrypoint()
