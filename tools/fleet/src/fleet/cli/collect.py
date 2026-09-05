"""CLI: ask the nodes which dispatches have finished, and close them out.

Usage:
    fleet-collect --config fleet.json
    fleet-collect --config fleet.json --run tools-fleet-1788556688

THE COMMAND THAT MAKES A RESULT ARRIVE. ``fleet-run`` returns as soon as the
suite is running, by design -- it launches through the node's task scheduler
precisely so the build outlives the ssh call. Something therefore has to go
back and ask, and this is it: for every dispatch the ledger still calls
running, it reads the node's recorded exit status, and for each that has one
it appends the closing row, emits ``passed`` or ``failed`` on the feed, and
releases the lease.

IT IS SAFE TO RUN AS OFTEN AS YOU LIKE. A run that has not finished is left
exactly as it was, and a run that has is closed once -- the second call finds
no live row for it. That is what makes the subscription loop a shell loop:

    while true; do
      fleet-collect --config fleet.json
      fleet-watch   --config fleet.json
      sleep 30
    done

IT EXITS 0 FOR A FAILING SUITE. The status of this command is whether
COLLECTION worked, not whether the work passed; a red suite is a result that
was successfully collected. Conflating the two would make a shell loop stop
on the first failing build, which is the one moment somebody wants the loop to
keep reporting. What the suite did is on the feed and in the ledger's
``outcome``.

A NODE THAT WILL NOT ANSWER STOPS THE COMMAND. There is no catching here: if
one node is unreachable the refusal propagates with its own code, rather than
this reporting the runs it did manage to collect and silently omitting a node
whose dispatches may have finished hours ago.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.errors import AppError, FleetErrorCode
from platform_core.logging import get_logger, setup_logging

from fleet.cli import _config
from fleet.contracts.ledger import LedgerEntry, is_live
from fleet.contracts.project import ProjectConfig
from fleet.contracts.workspace import require_node, require_project
from fleet.core import collect, dispatch, records

_log = get_logger(__name__)

RUN_FLAG = "--run"

_FLAGS = (_config.CONFIG_FLAG, RUN_FLAG)


def live_rows(loaded: _config.LoadedWorkspace, *, run_id: str | None) -> tuple[LedgerEntry, ...]:
    """Find the dispatches the ledger still calls running.

    The reduction to one row per dispatch lives in
    :func:`fleet.core.records.latest_rows`, which is also what the capacity
    check counts through. Spelling it a second time here is how the two came
    to disagree once already: the capacity check counted superseded rows and
    declared a node permanently full.

    Args:
        loaded: The workspace and its resolved record paths.
        run_id: Consider only this dispatch, or None for all of them.

    Returns:
        One row per still-running dispatch, in the order they were started.
    """
    return tuple(
        row
        for row in records.latest_rows(loaded.ledger)
        if is_live(row) and (run_id is None or row["run_id"] == run_id)
    )


def collect_one(loaded: _config.LoadedWorkspace, row: LedgerEntry) -> str:
    """Close one dispatch out if its node says it has finished.

    Args:
        loaded: The workspace and its resolved record paths.
        row: The running row.

    Returns:
        One line saying what happened, for the log.

    Raises:
        AppError: With ``WORKSPACE_NODE_UNKNOWN`` if the node has since left
            the workspace, ``WORKSPACE_PROJECT_UNKNOWN`` if its project has,
            a transport code if the node cannot be reached,
            ``RUN_RESULT_UNREADABLE`` if it answered with something that is
            not a status and a timestamp, or ``LEASE_NOT_HELD`` if the build
            was still writing after its lease had lapsed -- which means
            another dispatch could have been admitted into the same
            environment, and is not something to record a tidy outcome over.
    """
    node = require_node(loaded.workspace, row["node"])
    result = collect.poll_result(node, run_id=row["run_id"])
    if result is None:
        return f"{row['run_id']}: still running on {row['node']}"

    plan = require_project(loaded.workspace, row["project"])
    if collect.outlived_its_lease(row, plan, finished_unix=result["finished_unix"]):
        raise lapsed_lease_refusal(row, plan, finished_unix=result["finished_unix"])

    exit_code = result["exit_code"]
    detail = collect.describe(node, run_id=row["run_id"], exit_code=exit_code)
    dispatch.finish(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        row=row,
        outcome=collect.outcome_for(exit_code),
        exit_code=exit_code,
        detail=detail,
    )
    return f"{row['run_id']}: {collect.outcome_for(exit_code)} -- {detail}"


def lapsed_lease_refusal(
    row: LedgerEntry, plan: ProjectConfig, *, finished_unix: int
) -> AppError[FleetErrorCode]:
    """Build the refusal for a run that was still going without a lease.

    Refused rather than closed quietly. While the lease was gone and the
    build was still writing, a second dispatch could have been admitted into
    the same environment, so this result may have been produced by two suites
    interfering -- and recording the exit status as though nothing happened
    would be the last chance anybody had to notice.

    Note this is NOT the same as collecting late. A run that finished inside
    its window is closed normally however long afterwards it is read.

    Args:
        row: The running row.
        plan: Its project's declaration, which sized the lease.
        finished_unix: When the node says the build ended.

    Returns:
        The error to raise.
    """
    deadline = collect.lease_deadline(row, plan)
    return AppError(
        FleetErrorCode.LEASE_NOT_HELD,
        f"{row['run_id']} was still running on {row['node']} {finished_unix - deadline}s after "
        f"its lease lapsed at {deadline}; the environment was unprotected while the build was "
        "still writing, so the result is not recorded. Close it deliberately with "
        f"fleet-cancel --run {row['run_id']} once you have established nothing else was "
        f"dispatched to {row['node']} in that window, and raise the project's "
        "expected_minutes so the next run's lease covers it",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Collect every finished dispatch.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 when every live dispatch was asked about, whatever the suites did.
        See the module docstring for why a failing build is not an error
        here.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing its value.
        AppError: As :func:`collect_one` describes.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)

    rows = live_rows(loaded, run_id=parsed.get(RUN_FLAG))
    if not rows:
        _log.info("no dispatch is running")
        return 0
    for row in rows:
        _log.info("%s", collect_one(loaded, row))
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="fleet-collect",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["collect_one", "entrypoint", "live_rows", "main"]


# Without this, `python -m fleet.cli.collect` imports the module, runs nothing
# and exits 0 -- which reads as "nothing has finished" and would leave every
# dispatch open while looking like a clean collection.
if __name__ == "__main__":
    entrypoint()
