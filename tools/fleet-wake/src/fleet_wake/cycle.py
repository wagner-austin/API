"""One poll: ledger to board to position record, in that order.

THE ORDER IS THE DELIVERY GUARANTEE. Announcements POST before position rows
are WRITTEN, so a crash between the two repeats a post on the next cycle
rather than losing one. At-least-once, with the position file as the mark.
The alternative -- record first, then post -- turns any transport failure into
a dispatch nobody is ever told about, which is precisely the silence this
bridge exists to remove.

AND NOTHING IS CAUGHT. A refused post ends the cycle with a non-zero exit for
the scheduler to record, and the position rows for that group are not written,
so the next cycle tries again. A bridge that swallowed the failure would write
its position anyway and never announce that work again -- reporting success
while doing the opposite of its job.

EVERYTHING LEDGER-SHAPED IS ``fleet``'s OWN MACHINERY: ``latest_rows``
collapses the append-only ledger to the current row per dispatch, and
``is_live`` decides terminality. Both carry reasoning this package must not
re-learn -- reading raw rows instead of current ones announces every dispatch
twice, once from its running row and once from its terminal one.
"""

from __future__ import annotations

import pathlib

from board_watch.config import load_credentials
from fleet.cli._config import LoadedWorkspace
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core import records
from platform_core.board import post_to_task
from platform_core.json_utils import load_json_str

from fleet_wake import _test_hooks
from fleet_wake.announce import announcements, terminal_unannounced
from fleet_wake.identity import IDENTITY, load_task_id
from fleet_wake.position import AnnouncedRun, append_announced, position_path, read_announced


def run_cycle(loaded: LoadedWorkspace) -> None:
    """Run one bridge cycle against one fleet workspace.

    Args:
        loaded: The decoded workspace and its resolved record paths.

    Raises:
        AppError: Configuration (missing credentials or task id) or transport
            (the board refused).
        JSONTypeError: A ledger row or position line that does not decode.
        OSError: The ledger or position file cannot be read or written.
    """
    credentials = load_credentials()
    task_id = load_task_id()

    ledger_path = loaded.ledger
    rows = records.latest_rows(ledger_path)
    if rows == ():
        _test_hooks.emit("ledger is empty; nothing has been dispatched from this machine")
        return

    marks = position_path(ledger_path)
    announced = read_announced(marks)
    fresh = terminal_unannounced(rows, announced)
    if fresh == ():
        _test_hooks.emit(f"{len(rows)} dispatch(es) recorded, none newly terminal")
        return

    for announcement in announcements(fresh):
        # CALLED DIRECTLY, WITH NO PACKAGE-LOCAL post_announcement IN FRONT OF
        # IT. There was one, and it did nothing but bind this package's HTTP
        # seam and identity into this same call from a single call site --
        # which is a wrapper, not a boundary. The two constants it bound are
        # right here and read as what they are.
        post_to_task(
            _test_hooks.http_post,
            credentials,
            IDENTITY,
            task_id=task_id,
            kind="note",
            body=announcement["body"],
        )
        _test_hooks.emit(f"posted {announcement['project']}: tagged @{announcement['agent']}")

    at = _test_hooks.now()
    for entry in fresh:
        append_announced(
            marks,
            AnnouncedRun(run_id=entry["run_id"], outcome=entry["outcome"], announced_unix=at),
        )
    _test_hooks.emit(
        f"cycle: {len(rows)} recorded, {len(fresh)} newly terminal, positions recorded"
    )


def load_workspace(config_path: pathlib.Path) -> LoadedWorkspace:
    """Read the fleet workspace this bridge announces for.

    THE SAME DOCUMENT EVERY ``fleet-*`` COMMAND READS, decoded by fleet's own
    decoder, so the bridge cannot disagree with the dispatcher about where the
    ledger is. Reading it here rather than taking a ledger path directly is
    what makes that impossible rather than merely unlikely.

    Args:
        config_path: Path to the workspace document.

    Returns:
        The decoded workspace and its resolved record paths.

    Raises:
        JSONTypeError: If the document is not a valid fleet workspace.
        OSError: If it cannot be read.
    """
    resolved = config_path.resolve()
    document = load_json_str(_test_hooks.read_text(resolved))
    return LoadedWorkspace(decode_fleet_workspace(document), resolved.parent)


__all__ = ["load_workspace", "run_cycle"]
