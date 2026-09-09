"""One poll: journal to board to position file.

THE ORDER IS THE DELIVERY GUARANTEE. The announcement POSTS before the
offset ADVANCES, so a crash between the two repeats a post on the next
cycle rather than losing one. At-least-once, with the position file as the
mark -- the family order, for the family reason.

AND NOTHING IS CAUGHT. A refused post ends the cycle with a non-zero exit
for the pump to record, and the offset is not written, so the next cycle
tries again. A bridge that swallowed the refusal would advance its offset
anyway and never announce those transitions again -- reporting success
while doing the opposite of its job.

PROGRESS-ONLY SLICES ADVANCE THE OFFSET WITHOUT A POST. That is the noise
budget doing its work, not a dropped event: requested/waiting/step lines
are folded into their hold's boundary line when one is present in the same
window, and consumed as counts-nobody-needed when the boundary arrives in
a later window (:mod:`lock_wake.announce` states the trade). The offset
still advances only AFTER the decision that there is nothing to post, and
that decision is pure, so nothing can be lost between them.
"""

from __future__ import annotations

import pathlib

from board_watch.config import load_credentials
from platform_core.board import post_to_task

from lock_wake import _test_hooks
from lock_wake.announce import announcement
from lock_wake.identity import IDENTITY, load_task_id
from lock_wake.journal import read_journal_slice
from lock_wake.position import position_path, read_offset, write_offset


def run_cycle(journal: pathlib.Path) -> None:
    """Run one bridge cycle against one journal.

    Args:
        journal: Path to ``.fleet-events.jsonl``, the lock wrapper's
            append-only record.

    Raises:
        AppError: Configuration (missing credentials or task id) or the
            board refusing a post.
        JSONTypeError: A journal line or position file that does not
            decode.
        InvalidJsonError: A journal line or position file that is not
            JSON at all.
        ValueError: A position pointing past the journal's end -- the
            journal was truncated or replaced, and the operator decides,
            not this reader.
        OSError: A journal or position file that cannot be read or
            written.
    """
    credentials = load_credentials()
    task_id = load_task_id()

    marks = position_path(journal)
    offset = read_offset(marks)
    journal_slice = read_journal_slice(journal, offset)
    events = journal_slice["events"]
    if events == ():
        _test_hooks.emit(f"journal quiet; offset {offset}")
        return

    post = announcement(events)
    if post is None:
        write_offset(marks, journal_slice["next_offset"])
        _test_hooks.emit(
            f"{len(events)} progress line(s), no boundary; offset {journal_slice['next_offset']}"
        )
        return

    post_to_task(
        _test_hooks.http_post,
        credentials,
        IDENTITY,
        task_id=task_id,
        kind="note",
        body=post["body"],
    )
    write_offset(marks, journal_slice["next_offset"])
    tagged = " ".join(f"@{agent}" for agent in post["agents"])
    _test_hooks.emit(
        f"posted {post['holds']} hold(s) from {len(events)} line(s)"
        + (f": tagged {tagged}" if tagged != "" else ": unaddressed, no agent recorded")
    )


__all__ = ["run_cycle"]
