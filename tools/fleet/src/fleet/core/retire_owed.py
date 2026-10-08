"""A settle's retire that its node did not answer, owed to a later pass (MCPs board task 8776b828).

A node runner's settle closes the queue job and finishes the ledger row
before it retires the run's directory (:mod:`fleet.cli.node_settle`), so the
session waiting on the row is not kept waiting on housekeeping. Until this
module that retire raised ``NODE_UNREACHABLE`` when the node's ssh timed
out, after both closes: no later pass retried it, since nothing was live
any more, so the run's export stayed on the node and its transcript never
reached ``<stage_root>/logs/<run_id>.log``, the path the verdict names; and
the raise ended the serve, so every other run the runner held waited for
the next start. lavender-wsl's ssh timed out its banner exchange at
02:53Z-02:55Z, 03:00Z and 04:05Z on 2026-10-07.

So the settle ATTEMPTS the retire (:func:`retire_or_owe`). A node that did
not answer is recorded as owing it, one ``owed`` line in the retire record
beside the ledger (:mod:`fleet.contracts.retire_record`), and the settle
returns as if it had retired. Every collect pass then sends this node's
owed retires (:func:`retire_owed`): an answered one is recorded
``retired`` and never sent again, and the node's orphaned virtualenvs,
which only the retire made orphans, are swept after it; one the node still
did not answer stays owed, and the rest of the node's wait with it.

A retire the node ANSWERED and failed is a fault, as it was: it is recorded
``failed`` and raised by name with the node's own error, and never sent
again, because a pass that resent it would end every serve after it the
same way.
"""

from __future__ import annotations

import pathlib
from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from platform_core.logging import get_logger

from fleet.contracts.node import NodeConfig
from fleet.contracts.retire_record import RetireRecord, RetireState
from fleet.core import _test_hooks, names, records, retire, venv_sweep

_log = get_logger(__name__)

#: The retire record's file name, beside the ledger
#: (:attr:`fleet.cli._config.LoadedWorkspace.retires`).
RETIRES_FILE: Final = "retires.jsonl"


def _retire(path: pathlib.Path, node: NodeConfig, *, alias: str, run_id: str, owed: bool) -> bool:
    """Send one run's retire and record what came of it.

    Args:
        path: The retire record file.
        node: The node the run ran on.
        alias: That node's workspace name.
        run_id: The run.
        owed: Whether the record already holds the run as owed, so an
            answered retire is recorded ``retired`` and a missed one is not
            recorded ``owed`` a second time.

    Returns:
        True when the node answered and the retire was done; False when the
        node did not answer and the run is owed.

    Raises:
        AppError: ``DISPATCH_FAILED`` with the node's own error when it
            answered and the retire failed there, recorded ``failed`` first.
    """
    failure = retire.attempt_retire_on_node(node, run_id=run_id)["failure"]
    if failure is None:
        if owed:
            retained = names.retained_log_path(node["stage_root"], run_id)
            records.append_retire(
                path,
                RetireRecord(
                    run_id=run_id,
                    node=alias,
                    state=RetireState.RETIRED,
                    at_unix=_test_hooks.now(),
                    detail=f"its transcript is kept at {retained}",
                ),
            )
            _log.info("retire %s on %s: done, owed since its settle", run_id, alias)
        return True
    if failure["code"] is FleetErrorCode.NODE_UNREACHABLE:
        if not owed:
            records.append_retire(
                path,
                RetireRecord(
                    run_id=run_id,
                    node=alias,
                    state=RetireState.OWED,
                    at_unix=_test_hooks.now(),
                    detail=failure["message"],
                ),
            )
        _log.info(
            "retire %s on %s: the node did not answer; owed to the next collect pass: %s",
            run_id,
            alias,
            failure["message"],
        )
        return False
    records.append_retire(
        path,
        RetireRecord(
            run_id=run_id,
            node=alias,
            state=RetireState.FAILED,
            at_unix=_test_hooks.now(),
            detail=failure["message"],
        ),
    )
    raise AppError(failure["code"], failure["message"])


def retire_or_owe(path: pathlib.Path, node: NodeConfig, *, alias: str, run_id: str) -> bool:
    """Retire a settled run's directory, or record it as owed when the node did not answer.

    Args:
        path: The retire record file.
        node: The node the run ran on.
        alias: That node's workspace name.
        run_id: The run, whose row and queue job are already closed.

    Returns:
        True when it was retired; False when the node did not answer and
        :func:`retire_owed` retires it at a later pass.

    Raises:
        AppError: ``DISPATCH_FAILED`` with the node's own error when it
            answered and the retire failed there. Not caught.
    """
    return _retire(path, node, alias=alias, run_id=run_id, owed=False)


def owed_on(path: pathlib.Path, *, alias: str) -> tuple[str, ...]:
    """The runs on one node whose retire is owed.

    Args:
        path: The retire record file.
        alias: The node's workspace name.

    Returns:
        Each run whose last line is ``owed``, in the order first recorded.

    Raises:
        JSONTypeError: When a line of the record does not decode.
    """
    latest: dict[str, RetireState] = {}
    for record in records.read_retires(path):
        if record["node"] == alias:
            latest[record["run_id"]] = record["state"]
    return tuple(run_id for run_id, state in latest.items() if state is RetireState.OWED)


def retire_owed(path: pathlib.Path, node: NodeConfig, *, alias: str) -> None:
    """Send every retire this node owes, then sweep its orphaned virtualenvs.

    Args:
        path: The retire record file.
        node: The node.
        alias: Its workspace name.

    Raises:
        AppError: ``DISPATCH_FAILED`` with the node's own error when it
            answered and a retire failed there, that run recorded ``failed``
            and never sent again; or a sweep the node answered and failed.
            Not caught.
    """
    retired = 0
    for run_id in owed_on(path, alias=alias):
        if not _retire(path, node, alias=alias, run_id=run_id, owed=True):
            # The node did not answer this one, so the rest wait for the
            # next pass rather than each spending its own timeout.
            break
        retired += 1
    if retired:
        venv_sweep.sweep_on_node(node)


__all__ = ["RETIRES_FILE", "owed_on", "retire_or_owe", "retire_owed"]
