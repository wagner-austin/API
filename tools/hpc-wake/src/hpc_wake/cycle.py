"""One poll: ledger to accounting to board to closures, in that order.

The order is the delivery guarantee. Announcements POST before closures are
WRITTEN, so a crash between the two repeats a post on the next cycle rather
than losing one -- at-least-once, with the closure file as the position.
That file is the same one ``hpc3-triage`` reads and writes, which is both
the point and the one stated limitation: a job that triage closes before
this bridge ever sees it terminal is closed unannounced. Triage is a human
running a command and reading its answer; the bridge exists for the jobs
nobody was watching.

Everything cluster- and ledger-shaped is ``hpc3``'s own machinery -- the
batched ``sacct`` call, the array base-id collapse, the aggregate-row
expansion, terminal-state classification -- because every one of those
carries a measured trap this package must not re-learn.
"""

from __future__ import annotations

import pathlib

from board_watch.config import load_credentials
from hpc3.contracts.array import base_job_ids
from hpc3.contracts.cluster import ClusterFacts
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.workspace import WorkspaceConnection
from hpc3.core import ledger
from hpc3.core.remote import run_remote_batched
from hpc3.core.status import parse_sacct_output, sacct_commands
from hpc3.core.triage import closures_for, open_entries
from platform_core.board import post_to_task

from hpc_wake import _test_hooks
from hpc_wake.announce import announcements
from hpc_wake.identity import IDENTITY, load_task_id


def run_cycle(connection: WorkspaceConnection, cluster: ClusterFacts) -> None:
    """Run one bridge cycle against one workspace.

    Args:
        connection: Where the cluster and the ledger are.
        cluster: The measured cluster the ledger's rows are validated
            against.

    Raises:
        AppError: Configuration (missing credentials or task id), transport
            (the board refused), cluster (``ssh``/``sacct`` failed), or the
            ledger-integrity code ``JOB_UNKNOWN_TO_LEDGER``. Nothing is
            caught: the scheduler that runs this sees a non-zero exit, and a
            bridge that swallowed its own failure would report the silence
            it exists to remove.
    """
    credentials = load_credentials()
    task_id = load_task_id()

    ledger_path = pathlib.Path(connection["ledger"])
    entries = ledger.read(ledger_path, cluster)
    if entries == []:
        _test_hooks.emit("ledger is empty; nothing has been submitted from this machine")
        return
    closures_path = ledger.closure_path(ledger_path)
    known = ledger.read_closures(closures_path)

    still_open = open_entries(entries, known)
    if still_open == []:
        _test_hooks.emit(f"{len(entries)} recorded, all closed; nothing to announce")
        return

    job_ids = base_job_ids([entry["job_id"] for entry in still_open])
    statuses = parse_sacct_output(
        run_remote_batched(connection["host"], sacct_commands(job_ids)), cluster
    )
    ended = closures_for(statuses, closed_at=_test_hooks.now_iso())
    # closures_for expands aggregate rows to every task id, including tasks
    # whose own closure is already written; announcing those again would
    # repeat old news on every cycle that sees the aggregate.
    fresh = [closure for closure in ended if closure["job_id"] not in known]
    if fresh == []:
        _test_hooks.emit(f"{len(still_open)} open job(s), none newly terminal")
        return

    entries_by_id: dict[str, LedgerEntry] = {entry["job_id"]: entry for entry in entries}
    for announcement in announcements(fresh, entries_by_id):
        # CALLED DIRECTLY. Until the 2026-09-06 lift this package's board.py
        # held the argument-building and the transport call, and was a real
        # module; moving that into platform_core.board left it binding two
        # local constants into one call from one call site, which is a
        # wrapper. Deleted rather than kept for symmetry with a sibling that
        # had the same husk for the same reason.
        post_to_task(
            _test_hooks.http_post,
            credentials,
            IDENTITY,
            task_id=task_id,
            kind="note",
            body=announcement["body"],
        )
        _test_hooks.emit(
            f"posted {announcement['project']}: "
            + (
                f"tagged @{announcement['submitter']}"
                if announcement["submitter"] != ""
                else "no submitter label on record"
            )
        )
    for closure in fresh:
        ledger.append_closure(closures_path, closure)
    _test_hooks.emit(
        f"cycle: {len(still_open)} open, {len(fresh)} newly terminal, closures recorded"
    )


__all__ = ["run_cycle"]
