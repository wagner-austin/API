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
from platform_core.board import post_to_task, register_service_session
from platform_core.journal_cursor import cursor_path, read_offset, write_offset

from lock_wake import _test_hooks
from lock_wake.announce import announcement
from lock_wake.identity import CURSOR_READER, HARNESS, IDENTITY, PURPOSE, load_task_id
from lock_wake.journal import read_journal_slice, require_check_rows


def run_cycle(journal: pathlib.Path, check_journal: pathlib.Path) -> None:
    """Run one bridge cycle against the fleet journal and the check journal.

    Both are read from their own positions, folded into at most one post,
    and both positions advance only after that post, or after the decision
    that there is nothing to post -- so the at-least-once order holds for
    each file.

    Args:
        journal: Path to ``.fleet-events.jsonl``, the lock wrapper's
            append-only record.
        check_journal: Path to ``.check-events.jsonl``, the MCPs check
            lock's record of finished ``make test`` runs, kept apart from the
            fleet journal because every checkout's older reader of that file
            refuses a kind it does not declare (MCPs board task ea2ea29c).

    Raises:
        AppError: Configuration (missing credentials or task id) or the
            board refusing a post.
        JSONTypeError: A journal line or position file that does not
            decode, or a check journal holding a lock transition.
        InvalidJsonError: A journal line or position file that is not
            JSON at all.
        ValueError: A position pointing past a journal's end -- the
            journal was truncated or replaced, and the operator decides,
            not this reader.
        OSError: A journal or position file that cannot be read or
            written.
    """
    credentials = load_credentials()
    task_id = load_task_id()

    marks = cursor_path(journal, CURSOR_READER)
    check_marks = cursor_path(check_journal, CURSOR_READER)
    journal_slice = read_journal_slice(
        journal, read_offset(_test_hooks.file_exists, _test_hooks.read_bytes, marks)
    )
    check_slice = read_journal_slice(
        check_journal, read_offset(_test_hooks.file_exists, _test_hooks.read_bytes, check_marks)
    )
    require_check_rows(check_slice["events"], check_journal)
    events = journal_slice["events"] + check_slice["events"]
    offsets = f"offsets {journal_slice['next_offset']} and {check_slice['next_offset']}"
    if events == ():
        _test_hooks.emit(f"journals quiet; {offsets}")
        return

    post = announcement(events)
    if post is None:
        write_offset(_test_hooks.write_text, marks, journal_slice["next_offset"])
        write_offset(_test_hooks.write_text, check_marks, check_slice["next_offset"])
        _test_hooks.emit(f"{len(events)} progress line(s), no boundary; {offsets}")
        return

    # THE LEDGER GATE COMES FIRST, and it is why all three bridges went
    # silent from 2026-09-16 to 2026-09-21: MCPs mig 514 refuses a write
    # from a session no ledger surface knows, a service session is exactly
    # one, and every tick since died on TASK_SESSION_UNLEDGERED with the
    # traceback going only to runs/cycle.log, which nothing reads. Placed
    # after the quiet returns above so a quiet tick stays quiet.
    register_service_session(
        _test_hooks.http_post,
        credentials,
        IDENTITY,
        harness=HARNESS,
        purpose=PURPOSE,
    )
    post_to_task(
        _test_hooks.http_post,
        credentials,
        IDENTITY,
        task_id=task_id,
        kind="note",
        body=post["body"],
    )
    write_offset(_test_hooks.write_text, marks, journal_slice["next_offset"])
    write_offset(_test_hooks.write_text, check_marks, check_slice["next_offset"])
    tagged = " ".join(f"@{agent}" for agent in post["agents"])
    _test_hooks.emit(
        f"posted {post['holds']} hold(s) and {post['checks']} check run(s) "
        f"from {len(events)} line(s)" + (f": tagged {tagged}" if tagged != "" else ": unaddressed")
    )


__all__ = ["run_cycle"]
