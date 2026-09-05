"""CLI: print the board mentions that have arrived since the last call.

Usage:
    board-watch --agent opus-example-0905
    board-watch --agent opus-example-0905 --room main --kind status_change
    board-watch --agent opus-example-0905 --state ./cursors

One line per matching event, oldest first, on standard output, then exit.
A Claude session subscribes by composing the loop in the shell::

    Monitor({
      command: "while true; do board-watch --agent <label>; sleep 45; done",
      description: "board mentions for <label>"
    })

IT DOES NOT FOLLOW, for the same reason ``fleet-watch`` does not: Monitor's
own guidance is that the polling loop belongs in the shell where its interval
and filter are visible at the call site, rather than hidden inside a command.
That also removes the clock from this package entirely.

THE FIRST CALL PRINTS NOTHING AND THAT IS CORRECT. With no cursor document
the watcher walks to the end of the feed, records that position, and reports
that it armed. A watcher that announced its whole backlog on startup would
wake a session for every mention it had already read, which is precisely the
noise a subscription is meant to remove.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args

from board_watch import _test_hooks
from board_watch.config import load_credentials
from board_watch.state import DEFAULT_STATE_DIRECTORY, WatchState, load_state, save_state
from board_watch.watch import MAX_LIMIT, SubscriptionSpec, format_notification, poll, prime

AGENT_FLAG = "--agent"
ROOM_FLAG = "--room"
KIND_FLAG = "--kind"
LIMIT_FLAG = "--limit"
STATE_FLAG = "--state"

ALLOWED_FLAGS = (AGENT_FLAG, ROOM_FLAG, KIND_FLAG, LIMIT_FLAG, STATE_FLAG)


def _spec(parsed: dict[str, str]) -> SubscriptionSpec:
    """Build the subscription from parsed flags.

    Args:
        parsed: The flag values.

    Returns:
        The subscription.

    Raises:
        ValueError: When ``--limit`` is not a positive integer within the
            board's page ceiling. Raised rather than clamped: a caller who
            asked for 500 rows has a wrong expectation, and silently giving
            them 200 leaves it wrong.
    """
    raw_limit = parsed.get(LIMIT_FLAG)
    limit = MAX_LIMIT if raw_limit is None else int(raw_limit)
    if limit < 1 or limit > MAX_LIMIT:
        raise ValueError(f"{LIMIT_FLAG} must be between 1 and {MAX_LIMIT}, got {limit}")
    return SubscriptionSpec(
        agent=cli_args.require_flag(parsed, AGENT_FLAG),
        room=parsed.get(ROOM_FLAG),
        kind=parsed.get(KIND_FLAG),
        limit=limit,
    )


def main(argv: Sequence[str]) -> int:
    """Print new mentions for one agent and record the new position.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        0 always. Every failure raises instead, so an unreachable board or a
        changed render contract ends the process non-zero and the subscriber
        sees it. A watcher that returned a status nobody reads would be
        indistinguishable from a quiet board.

    Raises:
        AppError: Any configuration, transport or contract failure.
        ValueError: An unusable flag value.
    """
    parsed = cli_args.parse_single_flags(argv, ALLOWED_FLAGS)
    spec = _spec(parsed)
    raw_directory = parsed.get(STATE_FLAG)
    directory = DEFAULT_STATE_DIRECTORY if raw_directory is None else pathlib.Path(raw_directory)
    credentials = load_credentials()

    existing = load_state(spec["agent"], directory)
    if existing is None:
        cursor = prime(credentials)
        save_state(WatchState(agent=spec["agent"], cursor=cursor), directory)
        _test_hooks.emit(
            f"BOARD WATCH armed for @{spec['agent']}; "
            "position set to the end of the feed, reporting new mentions only"
        )
        return 0

    page, moved = poll(credentials, spec, existing["cursor"])
    for event in page["events"]:
        _test_hooks.emit(format_notification(event))
    save_state(WatchState(agent=spec["agent"], cursor=moved), directory)
    return 0


def entrypoint() -> None:
    """Console-script wrapper.

    Raises:
        SystemExit: Always, carrying :func:`main`'s status.
    """
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    entrypoint()
