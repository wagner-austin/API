"""One poll: ledger to accounting to pending to board to closures, in order.

The order is the delivery guarantee. Newly terminal jobs are made durable in
the PENDING record before anything is announced, announcements POST before
closures are WRITTEN, so a crash anywhere repeats a post on the next cycle
rather than losing one -- at-least-once, with the closure file as the
position.

WHAT A CYCLE NO LONGER DOES IS ANNOUNCE EVERYTHING IT SEES. Endings wait in
the pending record until their group settles (:mod:`hpc_wake.settling`),
because grouping only within one cycle made the post rate equal to the poll
rate: measured 2026-09-07, 116 posts in 24 hours at a 180-second median gap,
47 of ~132 carrying a single job. The poll can stay as frequent as the
operator likes; the board no longer pays for it.
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

from hpc_wake import _test_hooks, pending
from hpc_wake.announce import announcements, group_key
from hpc_wake.identity import IDENTITY, load_task_id
from hpc_wake.pending import PendingClosure
from hpc_wake.settling import partition_ripe


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

    # PRUNE THE PENDING RECORD HERE, above every early return, so that one
    # invariant holds on every path out of this function: the pending file
    # never names a job the closure file already holds.
    #
    # The crash window this covers is real -- die between the closure write
    # and the pending rewrite at the bottom and the ending sits in both
    # files. Filtering it only at the settling step below was enough to stop
    # it being announced twice, which is the property that matters, but it
    # left the stale record on disk forever once the ledger fully closed:
    # `still_open == []` returns before settling is ever reached. A record
    # that can never leave is a slow leak, and it was a test asserting the
    # file empties that found it rather than any reasoning about the code.
    #
    # Written only when something was actually pruned. An unconditional
    # write would create a pending file on every quiet cycle of a bridge
    # that has nothing pending, which is a new file where there was none.
    pending_file = pending.pending_path(ledger_path)
    on_disk = pending.read_pending(pending_file)
    waiting = [record for record in on_disk if record["closure"]["job_id"] not in known]
    if len(waiting) != len(on_disk):
        pending.write_pending(pending_file, waiting)

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
    entries_by_id: dict[str, LedgerEntry] = {entry["job_id"]: entry for entry in entries}

    # THE SETTLING STEP, and the order below is the delivery guarantee.
    #
    # Arrivals are made durable BEFORE anything is announced, and announced
    # BEFORE their closures are written. That leaves exactly two crash
    # windows and both are safe: die after the pending write and the
    # endings are announced next cycle; die after the announce and the
    # closure write repeats a post rather than losing one. At-least-once,
    # unchanged from before this step existed -- the position marker is
    # still the closure file.
    #
    # THERE IS NO `if fresh == []: return` ABOVE THIS, and that is load
    # bearing rather than an omission. The quiet rule fires precisely when
    # nothing new arrives, so a cycle that returned early on an empty
    # `fresh` could never announce a settled group -- the rule that does all
    # the work would have been unreachable, and the bridge would hold every
    # trickling array until MAX_HOLD_SECONDS instead. A cycle now returns
    # early only when there is nothing waiting AT ALL.
    #
    # ``waiting`` was read and pruned above, before the early returns, so
    # that the pending file never names an already-closed job on ANY path
    # out of this function.
    waiting_ids = {record["closure"]["job_id"] for record in waiting}
    now_epoch = _test_hooks.now_epoch()
    arrivals = [
        PendingClosure(closure=closure, observed_epoch=now_epoch)
        for closure in fresh
        if closure["job_id"] not in waiting_ids
    ]
    everything_waiting = [*waiting, *arrivals]
    if everything_waiting == []:
        _test_hooks.emit(f"{len(still_open)} open job(s), none newly terminal and none waiting")
        return
    pending.write_pending(pending_file, everything_waiting)

    keys = {job_id: group_key(entry) for job_id, entry in entries_by_id.items()}
    ripe, holding = partition_ripe(everything_waiting, keys, now_epoch)
    if ripe == []:
        _test_hooks.emit(
            f"{len(everything_waiting)} ending(s) waiting, none settled; "
            f"{len(arrivals)} arrived this cycle"
        )
        return

    for announcement in announcements([record["closure"] for record in ripe], entries_by_id):
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
    for record in ripe:
        ledger.append_closure(closures_path, record["closure"])
    pending.write_pending(pending_file, holding)
    _test_hooks.emit(
        f"cycle: {len(still_open)} open, {len(arrivals)} newly terminal, "
        f"{len(ripe)} announced, {len(holding)} still settling"
    )


__all__ = ["run_cycle"]
