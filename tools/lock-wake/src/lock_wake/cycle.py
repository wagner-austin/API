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

TWO CYCLES, ONE PUBLISH. :func:`run_cycle` reads the hub's two local
journals; :func:`run_remote_cycle` reads one journal on another host over
ssh (MCPs board task 03590bf9: every deploy takes its lock on diphtheria).
Each is its own pump row with its own cursor, and both hand their slices to
:func:`_publish`, so the order above is written once.
"""

from __future__ import annotations

import pathlib
from typing import Final

from board_watch.config import load_credentials
from platform_core.board import post_to_task, register_service_session
from platform_core.journal_cursor import cursor_path, read_offset, write_offset
from platform_core.mcp_client import McpCredentials
from typing_extensions import TypedDict

from lock_wake import _test_hooks
from lock_wake.announce import announcement
from lock_wake.identity import CURSOR_READER, HARNESS, IDENTITY, PURPOSE, load_task_id
from lock_wake.journal import JournalSlice, read_journal_slice, require_check_rows
from lock_wake.remote import RemoteJournal, read_remote_slice, remote_cursor_path

#: The name the hub's own journals are announced under. The local cycle runs
#: only on the hub, from the pump's row there.
HUB: Final = "hub"


class _Stream(TypedDict):
    """One journal's slice and the cursor it advances.

    Attributes:
        journal_slice: What this tick read from the journal.
        cursor: The position file the slice's next offset is written to.
    """

    journal_slice: JournalSlice
    cursor: pathlib.Path


def run_cycle(journal: pathlib.Path, check_journal: pathlib.Path) -> None:
    """Run one bridge cycle against the hub's fleet journal and check journal.

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
    journal_slice = read_journal_slice(journal, _offset(marks))
    check_slice = read_journal_slice(check_journal, _offset(check_marks))
    require_check_rows(check_slice["events"], check_journal)
    _publish(
        credentials,
        task_id,
        HUB,
        (
            _Stream(journal_slice=journal_slice, cursor=marks),
            _Stream(journal_slice=check_slice, cursor=check_marks),
        ),
    )


def run_remote_cycle(journal: RemoteJournal, cursor_dir: pathlib.Path) -> None:
    """Run one bridge cycle against a fleet journal on another host.

    Args:
        journal: The host and the journal's path there.
        cursor_dir: The local directory holding this reader's position in
            it (:func:`lock_wake.remote.remote_cursor_path`).

    Raises:
        AppError: Configuration, the board refusing a post, or the ssh read
            failing (:func:`lock_wake.remote.read_remote_slice`).
        JSONTypeError: A journal line or position file that does not decode.
        InvalidJsonError: A journal line or position file that is not JSON.
        ValueError: A position past the remote journal's end.
        OSError: A position file that cannot be read or written.
        subprocess.TimeoutExpired: The ssh outlived its deadline.
    """
    credentials = load_credentials()
    task_id = load_task_id()
    marks = remote_cursor_path(cursor_dir, journal)
    journal_slice = read_remote_slice(journal, _offset(marks))
    _publish(
        credentials,
        task_id,
        journal["host"],
        (_Stream(journal_slice=journal_slice, cursor=marks),),
    )


def _offset(marks: pathlib.Path) -> int:
    """Read one position file through this package's file seams.

    Args:
        marks: The position file.

    Returns:
        The recorded offset, 0 when never written.
    """
    return read_offset(_test_hooks.file_exists, _test_hooks.read_bytes, marks)


def _publish(
    credentials: McpCredentials, task_id: str, host: str, streams: tuple[_Stream, ...]
) -> None:
    """Post one tick's boundaries, then advance every stream's cursor.

    Args:
        credentials: The board's address and secrets, loaded before any
            journal was read so a misconfigured pump fails before ssh.
        task_id: The standing task the post lands in.
        host: The machine the journals belong to, for the post and the
            report line.
        streams: Every journal this tick read, with its cursor.

    Raises:
        AppError: The board refusing a post; no cursor moves.
        OSError: A position file that cannot be written.
    """
    events = tuple(event for stream in streams for event in stream["journal_slice"]["events"])
    offsets = ", ".join(str(stream["journal_slice"]["next_offset"]) for stream in streams)
    if events == ():
        _test_hooks.emit(f"{host} journals quiet; offsets {offsets}")
        return

    post = announcement(events, host)
    if post is None:
        _advance(streams)
        _test_hooks.emit(f"{host}: {len(events)} progress line(s), no boundary; offsets {offsets}")
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
    _advance(streams)
    tagged = " ".join(f"@{agent}" for agent in post["agents"])
    _test_hooks.emit(
        f"{host}: posted {post['holds']} hold(s) and {post['checks']} check run(s) "
        f"from {len(events)} line(s)" + (f": tagged {tagged}" if tagged != "" else ": unaddressed")
    )


def _advance(streams: tuple[_Stream, ...]) -> None:
    """Write every stream's next offset to its cursor.

    Args:
        streams: Every journal this tick read, with its cursor.

    Raises:
        OSError: A position file that cannot be written.
    """
    for stream in streams:
        write_offset(
            _test_hooks.write_text, stream["cursor"], stream["journal_slice"]["next_offset"]
        )


__all__ = ["HUB", "run_cycle", "run_remote_cycle"]
