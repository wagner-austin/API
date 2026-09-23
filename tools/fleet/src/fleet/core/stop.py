"""Stopping a dispatch on its node, and closing it out as stopped.

ONE STOP, THREE CALLERS. ``fleet-cancel`` stops a run somebody named; a node
runner stops a run whose queue job was cancelled under it, and one still going
past its lease (MCPs board task fd5cabfa). All three need the same two facts
to hold afterwards: nothing the build started is still running on the node,
and the ledger no longer counts the run against that node's capacity. Those
used to be spelled once, inside the cancel command, where a runner could not
reach them; a runner that re-spelled them would be one edit away from a stop
that ended the task but not the suite, which is the defect this module's
dialect half exists to close (:meth:`fleet.core.dialect.Dialect.stop_script`).

THE NODE IS STOPPED BEFORE THE ROW IS CLOSED. A row closed first and a stop
that then failed would free the node's budget while the suite still held it,
and the next capacity check would admit a second dispatch onto a node that
was already full. In this order a failed stop propagates with the row still
live, which is the state that matches the node.
"""

from __future__ import annotations

import pathlib

from fleet.contracts.ledger import LedgerEntry, LedgerOutcome
from fleet.contracts.node import NodeConfig
from fleet.core import dialect, dispatch, names, remote


def stop_on_node(node: NodeConfig, *, run_id: str) -> None:
    """End one dispatch's build on its node, every process of it.

    The task or unit is named by :func:`fleet.core.names.task_name`, the
    same function the dispatch launched it with, and the script is named
    after the run so two stops at once cannot overwrite each other's.

    Args:
        node: The node it was dispatched to.
        run_id: The dispatch.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            transport, the latter including a verified build process that
            ``taskkill`` failed to end.
    """
    spoken = dialect.for_platform(node["platform"])
    remote.run_script(
        node["host"],
        spoken.script_path(node["stage_root"], names.stop_stem(run_id)),
        spoken.stop_script(
            target=names.dispatch_directory(node["stage_root"], run_id), run_id=run_id
        ),
        platform=node["platform"],
    )


def stop_and_finish(
    loaded_leases: pathlib.Path,
    loaded_ledger: pathlib.Path,
    loaded_feed: pathlib.Path,
    *,
    node: NodeConfig,
    row: LedgerEntry,
    outcome: LedgerOutcome,
    exit_code: int,
    detail: str,
) -> LedgerEntry:
    """Stop a dispatch on its node, then close its row, emit it, free its lease.

    Args:
        loaded_leases: The lease file.
        loaded_ledger: The ledger file.
        loaded_feed: The feed file.
        node: The node it was dispatched to.
        row: The running row this supersedes.
        outcome: How the dispatch ended.
        exit_code: The recipe's status, or
            :data:`~fleet.contracts.ledger.NO_EXIT_CODE` when there was none.
        detail: What to say about it, on the ledger and the feed alike.

    Returns:
        The closing row.

    Raises:
        AppError: As :func:`stop_on_node` describes, with the row still live.
    """
    stop_on_node(node, run_id=row["run_id"])
    return dispatch.finish(
        loaded_leases,
        loaded_ledger,
        loaded_feed,
        row=row,
        outcome=outcome,
        exit_code=exit_code,
        detail=detail,
    )


__all__ = ["stop_and_finish", "stop_on_node"]
