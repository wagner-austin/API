"""Every run's directory is owed a retire from its lease until one is done.

MCPs board tasks 8776b828 and 3d9eec0d. A dispatch makes
``<stage_root>/<run_id>`` and its staging directory on its node after it
takes its lease (:mod:`fleet.core.dispatch`), and only a
retire removes them (:mod:`fleet.core.retire`). So the debt starts at the
lease: :func:`owe` records the run ``owed`` in the retire record beside the
ledger (:mod:`fleet.contracts.retire_record`) the moment the lease is held
(:func:`fleet.core.run_lease.take`), and every retire that is done records
it ``retired`` (:func:`retire_or_owe`, :func:`retire_or_raise`). It is the
record's one writer of ``owed``: a retire the node did not answer leaves the
run owed and writes nothing, so a node down for hours adds no line a pass.

WHY AT THE LEASE AND NOT AT THE SETTLE. Until 2026-10-18 a run was recorded
only when its settle's retire was not answered, so a run that never settled
was recorded nowhere and nothing retired it. Measured 2026-10-18T13:06Z
(MCPs board task 3d9eec0d): 12 run directories on six nodes, each from a run
the hub's feed ends at ``refused`` (``NODE_UNREACHABLE`` while staging,
``STAGE_DIGEST_MISMATCH`` of a companion) or at ``leased`` or ``staged``
when its runner ended, and no line in the record for any of them.

Every collect pass sends this node's owed retires (:func:`retire_owed`),
except a run still live: one whose lease is held, which a launch is staging
or a build is running under, or whose ledger row is still running, which a
settle or a stop retires before or after its close. An answered one is
recorded ``retired`` and never sent again, and the node's unread cache
generations, which only the retire made unread, are removed after it; one
the node did not answer stays owed, and the rest of the node's wait with it.
A refused launch has given its lease back (:func:`fleet.core.run_lease.abandon`)
and wrote no ledger row, and a launch whose runner ended holds its lease
until it lapses, so the next pass after either retires it as it would a
settled run whose node missed the retire (lavender-wsl's ssh timed out its
banner exchange at 02:53Z-02:55Z, 03:00Z and 04:05Z on 2026-10-07, MCPs
board task 8776b828).

ONE RUN'S RETIRE IS SENT IN ITS OWN TURN, held across the hub's runners
(:func:`_turn`), and not sent at all when another turn already recorded it
done: a settle frees its lease before it retires, so a collect pass can find
the same run owed and no longer live while the settle is still retiring it.

A retire the node ANSWERED and failed is a fault: it is recorded ``failed``
and raised by name with the node's own error, and never sent again, because
a pass that resent it would end every serve after it the same way.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
from typing import Final

from platform_core.errors import AppError, FleetErrorCode
from platform_core.logging import get_logger

from fleet.contracts.ledger import is_live
from fleet.contracts.node import NodeConfig
from fleet.contracts.retire_record import RetireRecord, RetireState
from fleet.core import (
    _test_hooks,
    cache_generations,
    file_turn,
    leases,
    names,
    records,
    remote,
    retire,
)

_log = get_logger(__name__)

#: The retire record's file name, beside the ledger
#: (:attr:`fleet.cli._config.LoadedWorkspace.retires`).
RETIRES_FILE: Final = "retires.jsonl"

#: Where each run's retire turn is held, beside the record.
RETIRE_TURNS_DIRECTORY: Final = "retire-turns"

#: The turn one run's retire is sent and recorded in. Inside it a runner
#: sends the retire script and runs it, two ssh calls of at most
#: :data:`fleet.core.remote.SSH_TIMEOUT_SECONDS` each, so a lock older than
#: three of them was left by a process that died holding it.
RETIRE_TURN: Final = file_turn.TurnBounds(
    what="retire turn",
    waited_for="another runner retiring the same run",
    stale_seconds=3 * remote.SSH_TIMEOUT_SECONDS,
    poll_seconds=1,
)


def _latest_state(path: pathlib.Path, run_id: str) -> RetireState | None:
    """The state of one run's last line in the record.

    Args:
        path: The retire record file.
        run_id: The run.

    Returns:
        The state, or None when the record holds no line for it.

    Raises:
        JSONTypeError: When a line of the record does not decode.
    """
    state: RetireState | None = None
    for record in records.read_retires(path):
        if record["run_id"] == run_id:
            state = record["state"]
    return state


def _turn(path: pathlib.Path, run_id: str) -> contextlib.AbstractContextManager[None]:
    """One run's retire turn, held across this hub's runners and their threads.

    A settle closes its row and frees its lease before it retires, so a
    collect pass reading the record in that span finds the run owed and no
    longer live: measured 2026-10-18T18:38Z (MCPs board task 3d9eec0d),
    serendipity's settle and a collect pass sent one run's retire at once
    and each failed, one walking a tree the other was removing, and so did
    pendragon's watch and its own collect pass. Every retire is sent in its
    run's turn and first reads whether one is already recorded done.

    Args:
        path: The retire record file.
        run_id: The run.

    Returns:
        The turn, held for the span of a ``with`` block.
    """
    return file_turn.held(
        path.parent / RETIRE_TURNS_DIRECTORY / f"{run_id}.lock",
        holder=f"fleet process {os.getpid()}",
        bounds=RETIRE_TURN,
    )


def _record(
    path: pathlib.Path, *, alias: str, run_id: str, state: RetireState, detail: str
) -> None:
    """Append one change in a run's retire, stamped now.

    Args:
        path: The retire record file.
        alias: The run's node's workspace name.
        run_id: The run.
        state: What happened.
        detail: Why, as :class:`~fleet.contracts.retire_record.RetireRecord` says.
    """
    records.append_retire(
        path,
        RetireRecord(
            run_id=run_id, node=alias, state=state, at_unix=_test_hooks.now(), detail=detail
        ),
    )


def owe(path: pathlib.Path, *, alias: str, run_id: str) -> None:
    """Record a run as owing its retire, once its lease is held and before anything is staged.

    Args:
        path: The retire record file.
        alias: The node's workspace name.
        run_id: The run the lease names.
    """
    _record(
        path,
        alias=alias,
        run_id=run_id,
        state=RetireState.OWED,
        detail="its lease is held, so its directory is staged on the node",
    )


def _retired(path: pathlib.Path, node: NodeConfig, *, alias: str, run_id: str) -> str:
    """Record a run whose retire the node answered and did.

    Args:
        path: The retire record file.
        node: The node the run ran on.
        alias: That node's workspace name.
        run_id: The run.

    Returns:
        Where its transcript is now kept on the node.
    """
    retained = names.retained_log_path(node["stage_root"], run_id)
    _record(
        path,
        alias=alias,
        run_id=run_id,
        state=RetireState.RETIRED,
        detail=f"its transcript is kept at {retained}",
    )
    return retained


def retire_or_raise(path: pathlib.Path, node: NodeConfig, *, alias: str, run_id: str) -> str:
    """Retire a run whose row is still live, raising when the retire was not done.

    For a stop and the hub's ``fleet-collect``, which retire before the row
    closes, so a failure propagates with the row still live and the next
    tick settles the run again; the run stays owed until then. Sent in the
    run's turn (:func:`_turn`), and not at all when another turn already
    recorded it ``retired``.

    Args:
        path: The retire record file.
        node: The node the run ran on.
        alias: That node's workspace name.
        run_id: The run.

    Returns:
        Where its transcript is now kept on the node.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from the
            transport, the latter carrying the node's own error when a file
            could not be moved or removed.
    """
    with _turn(path, run_id):
        if _latest_state(path, run_id) is RetireState.RETIRED:
            return names.retained_log_path(node["stage_root"], run_id)
        failure = retire.attempt_retire_on_node(node, run_id=run_id)["failure"]
        if failure is not None:
            raise AppError(failure["code"], failure["message"])
        return _retired(path, node, alias=alias, run_id=run_id)


def retire_or_owe(path: pathlib.Path, node: NodeConfig, *, alias: str, run_id: str) -> bool:
    """Retire a run whose row is closed, or leave it owed when the node did not answer.

    For a node runner's settle, which retires after the queue close and the
    row's, and for :func:`retire_owed`. The run has been owed since its
    lease (:func:`owe`), so a retire the node did not answer writes nothing.
    Sent in the run's turn (:func:`_turn`), and not at all when another
    turn already recorded it ``retired`` or ``failed``.

    Args:
        path: The retire record file.
        node: The node the run ran on.
        alias: That node's workspace name.
        run_id: The run.

    Returns:
        True when the run owes nothing more: retired here, or ended by
        another turn; False when the node did not answer and
        :func:`retire_owed` retires it at a later pass.

    Raises:
        AppError: ``DISPATCH_FAILED`` with the node's own error when it
            answered and the retire failed there, recorded ``failed`` first.
            Not caught.
    """
    with _turn(path, run_id):
        ended = _latest_state(path, run_id)
        if ended is RetireState.RETIRED or ended is RetireState.FAILED:
            _log.info("retire %s on %s: already %s in another turn", run_id, alias, ended)
            return True
        failure = retire.attempt_retire_on_node(node, run_id=run_id)["failure"]
        if failure is None:
            _retired(path, node, alias=alias, run_id=run_id)
            _log.info("retire %s on %s: done", run_id, alias)
            return True
        if failure["code"] is FleetErrorCode.NODE_UNREACHABLE:
            _log.info(
                "retire %s on %s: the node did not answer; owed to the next collect pass: %s",
                run_id,
                alias,
                failure["message"],
            )
            return False
        _record(
            path, alias=alias, run_id=run_id, state=RetireState.FAILED, detail=failure["message"]
        )
        raise AppError(failure["code"], failure["message"])


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


def _live(leases_path: pathlib.Path, ledger_path: pathlib.Path) -> frozenset[str]:
    """The runs a retire must not touch yet: a held lease, or a running ledger row.

    Args:
        leases_path: The lease file.
        ledger_path: The ledger file.

    Returns:
        Their run ids.
    """
    held = leases.held_leases(leases_path, now_unix=_test_hooks.now())
    running = (row for row in records.latest_rows(ledger_path) if is_live(row))
    return frozenset(lease["run_id"] for lease in held) | {row["run_id"] for row in running}


def retire_owed(
    path: pathlib.Path,
    node: NodeConfig,
    *,
    alias: str,
    leases_path: pathlib.Path,
    ledger_path: pathlib.Path,
) -> None:
    """Send every retire this node owes for a run no longer live, then remove
    its unread cache generations.

    Args:
        path: The retire record file.
        node: The node.
        alias: Its workspace name.
        leases_path: The lease file, whose held leases name runs still
            launching or running.
        ledger_path: The ledger file, whose running rows name runs a settle
            or a stop retires.

    Raises:
        AppError: ``DISPATCH_FAILED`` with the node's own error when it
            answered and a retire failed there, that run recorded ``failed``
            and never sent again; or a removal the node answered and failed.
            Not caught.
    """
    owed = owed_on(path, alias=alias)
    if not owed:
        return
    live = _live(leases_path, ledger_path)
    retired = 0
    for run_id in owed:
        if run_id in live:
            continue
        if not retire_or_owe(path, node, alias=alias, run_id=run_id):
            # The node did not answer this one, so the rest wait for the
            # next pass rather than each spending its own timeout.
            break
        retired += 1
    if retired:
        cache_generations.remove_unused_on_node(node)


__all__ = ["RETIRES_FILE", "owe", "owed_on", "retire_or_owe", "retire_or_raise", "retire_owed"]
