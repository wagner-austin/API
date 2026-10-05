"""One poll: health journal to board to position file.

THE ORDER IS THE DELIVERY GUARANTEE. The post goes out before the offset
advances, so a crash between the two repeats a post on the next cycle
rather than losing one: at-least-once, the family order.

ONE POST PER CYCLE, every unread line in it. The audit writes a line only
when something changed, so a cycle usually has none or one; when the pump
was down across several audits, one post carrying all of them in order
keeps the delivery atomic -- a second post failing after a first succeeded
would re-post the first on the next cycle, where one post either lands
whole or not at all.

THEN THE OPERATOR ASKS (MCPs board task 1acbf53e): the audit's statement
beside the journal is stated to the board's writer whenever it changed,
:mod:`fleet_health_wake.asks`.

AND NOTHING IS CAUGHT. A refused post ends the cycle with a non-zero exit
for the pump to record, and the offset is not written, so the next cycle
tries again.
"""

from __future__ import annotations

import pathlib

from board_watch.config import load_credentials
from platform_core.board import post_to_task, register_service_session
from platform_core.journal_cursor import cursor_path, read_offset, write_offset
from platform_core.mcp_client import McpCredentials

from fleet_health_wake import _test_hooks
from fleet_health_wake.asks import OPERATOR_ASKS_FILENAME, state_asks
from fleet_health_wake.identity import CURSOR_READER, HARNESS, IDENTITY, PURPOSE, load_task_id
from fleet_health_wake.journal import HealthEvent, read_health_slice


def post_body(events: tuple[HealthEvent, ...]) -> str:
    """The one note a cycle posts: each line's body, in journal order.

    Args:
        events: The unread lines; at least one.

    Returns:
        The bodies separated by a blank line, since each already opens with
        its own summary line naming when and what.
    """
    return "\n\n".join(event["body"] for event in events)


def run_cycle(journal: pathlib.Path) -> None:
    """Run one bridge cycle: the health journal's unread lines, then the operator asks.

    The asks file is the journal's sibling in ``fleet-mcp/state``, where the
    audit writes both (MCPs ``fleet-mcp/src/operator-asks.ts``), and is
    stated after the journal so a refused post leaves both for the next cycle.

    Args:
        journal: Path to ``health-events.jsonl`` in MCPs ``fleet-mcp/state``.

    Raises:
        AppError: Configuration (missing credentials or task id) or the
            board refusing a post or a statement.
        JSONTypeError: A journal line, position, statement or marker file
            that does not decode.
        InvalidJsonError: A journal line, position or statement file that is
            not JSON.
        ValueError: A position into an absent journal or past its end --
            the journal was deleted, truncated or replaced, and the
            operator decides, not this reader.
        OSError: A file that cannot be read or written.
    """
    credentials = load_credentials()
    post_health(journal, credentials, load_task_id())
    state_asks(journal.parent / OPERATOR_ASKS_FILENAME, credentials)


def post_health(journal: pathlib.Path, credentials: McpCredentials, task_id: str) -> None:
    """Post the health journal's unread lines as one note, then advance the offset.

    Args:
        journal: Path to ``health-events.jsonl``.
        credentials: Endpoint and both board secrets.
        task_id: The standing task the note lands in.

    Raises:
        AppError: The board refusing the checkin or the note.
        JSONTypeError: A journal line or position file that does not decode.
        InvalidJsonError: A journal line or position file that is not JSON.
        ValueError: A position into an absent journal or past its end.
        OSError: A journal or position file that cannot be read or written.
    """
    marks = cursor_path(journal, CURSOR_READER)
    health = read_health_slice(
        journal, read_offset(_test_hooks.file_exists, _test_hooks.read_bytes, marks)
    )
    if health["events"] == ():
        _test_hooks.emit(f"health journal quiet; offset {health['next_offset']}")
        return

    # The ledger gate first: the board refuses a write from a session no
    # ledger surface knows (MCPs mig 514), and a service session is one.
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
        body=post_body(health["events"]),
    )
    write_offset(_test_hooks.write_text, marks, health["next_offset"])
    kinds = ", ".join(event["kind"] for event in health["events"])
    _test_hooks.emit(
        f"posted {len(health['events'])} health line(s) ({kinds}); offset {health['next_offset']}"
    )


__all__ = ["post_body", "post_health", "run_cycle"]
