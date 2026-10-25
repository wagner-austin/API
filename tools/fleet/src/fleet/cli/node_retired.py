"""Settle a run whose project the rolled workspace no longer declares (MCPs board task 4ec65b8f).

A roll that removes a project while a node runner holds a live run of it
left that run with nothing to judge it by. :func:`fleet.cli.node_collect.collect_one_job`
read the project's declaration before it polled, so the run's row raised
``WORKSPACE_PROJECT_UNKNOWN``, the collect pass ended there, and the runner
claimed nothing until the run was settled by some other means. Measured
2026-10-25: loki's node runner claimed job 45dc05d6 for MCPs/execution at
18:57:05Z, the roll of API 48bbcd716 (MCPs board task 5b040fdb) removed that
project at about 18:58Z, the build ended exit 0 at 19:00:47Z, and from
19:02:55Z every tick of fleet-node-loki on diphtheria ended TICK EXIT 2 on
that row, while the only windows suite of the roll that followed waited
queued behind it.

THE LEASE IS NOT JUDGED. The declaration is read for one thing here, the
lease (:func:`fleet.core.collect.lease_deadline`,
:func:`fleet.core.collect.outlived_its_lease`), which exists because past it
a second dispatch could be admitted into the run's environment. A project
the rolled workspace no longer declares admits no dispatch, so that question
has no subject, and the run is settled on what it did:

* FINISHED. It closes as any finished run does
  (:func:`fleet.cli.node_settle.settle`), with its own exit status, and its
  row's detail opens ``WORKSPACE_PROJECT_UNKNOWN`` naming the project.
* STILL RUNNING. Nothing will judge it when it ends, so its job is given
  back to the queue (:func:`fleet.cli.job_close.give_back`), which invents no
  exit status, and then its process tree is ended and its row closed lost
  (:func:`fleet.core.stop.stop_and_finish`). The queue goes first: a stop
  that fails leaves a live row whose job has left this runner, which the
  next pass's :func:`fleet.cli.node_lost.stop_lost` stops.
"""

from __future__ import annotations

from platform_core.error_codes_fleet import FleetErrorCode
from platform_core.json_utils import JSONObject
from platform_core.mcp_client import McpCredentials

from fleet.cli import _config, job_close
from fleet.cli.node_collected import Collected, CollectOutcome
from fleet.cli.node_settle import settle
from fleet.contracts.dispatch import DispatchJob, encode_job_line
from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.core import collect, stop
from fleet.core.collect import RunResult


def is_retired(loaded: _config.LoadedWorkspace, row: LedgerEntry) -> bool:
    """Whether the rolled workspace no longer declares a live row's project.

    Args:
        loaded: The workspace and its resolved record paths.
        row: A live ledger row.

    Returns:
        True when the workspace's projects do not name the row's.
    """
    return row["project"] not in loaded.workspace["projects"]


def settle_retired(
    loaded: _config.LoadedWorkspace,
    credentials: McpCredentials,
    board: McpCredentials,
    job: DispatchJob,
    identity: JSONObject,
    *,
    row: LedgerEntry,
    node: NodeConfig,
    result: RunResult | None,
) -> Collected:
    """Close a retired project's run with its own exit status, or give it back and stop it.

    Args:
        loaded: The workspace and its resolved record paths.
        credentials: The queue's endpoint and headers.
        board: The board's endpoint and headers, for the verdict.
        job: The queue job this runner holds.
        identity: This runner's identity arguments.
        row: The run's live ledger row, whose project :func:`is_retired`.
        node: The node it ran on, which answered the read of its result.
        result: How it ended, or None while it is still going.

    Returns:
        What :func:`fleet.cli.node_settle.settle` did for a finished run, or
        :attr:`~fleet.cli.node_collected.CollectOutcome.SETTLED` with the
        queue's answer for one given back.

    Raises:
        AppError: As :func:`fleet.cli.node_settle.settle` raises; a queue
            failure from the give-back, with nothing on this machine changed;
            or a node failure from the stop, which leaves the row live.
    """
    retired = (
        f"{FleetErrorCode.WORKSPACE_PROJECT_UNKNOWN.value}: {row['project']} is not in the "
        "rolled workspace, so its lease was not judged"
    )
    if result is not None:
        exit_code = result["exit_code"]
        described = collect.describe(node, run_id=row["run_id"], exit_code=exit_code)
        return settle(
            loaded,
            credentials,
            board,
            job,
            identity,
            row=row,
            node=node,
            exit_code=exit_code,
            ended_unix=result["finished_unix"],
            detail=f"{retired}; {described}",
            stopped=None,
        )
    detail = (
        f"{retired}; {row['run_id']} was still running on {row['node']}, so its job was given "
        "back and the runner ended its process tree"
    )
    answered = job_close.give_back(credentials, job, identity, detail=detail)
    stop.stop_and_finish(
        loaded.leases,
        loaded.ledger,
        loaded.feed,
        loaded.retires,
        node=node,
        row=row,
        outcome=LedgerOutcome.LOST,
        exit_code=NO_EXIT_CODE,
        detail=detail,
    )
    return Collected(
        outcome=CollectOutcome.SETTLED,
        line=f"{encode_job_line(job)}: {detail}; the queue left it {answered['status'].value}",
    )


__all__ = ["is_retired", "settle_retired"]
