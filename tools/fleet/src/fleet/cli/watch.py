"""CLI: the event stream a session subscribes to.

Usage:
    fleet-watch --config runs/fleet.json
    fleet-watch --config runs/fleet.json --run <run-id>

One line per event, oldest first, on standard output. That shape is the whole
design: a Claude session subscribes with
``Monitor({command: "fleet-watch --config runs/fleet.json"})`` and every line
becomes a notification, with no broker, no port, and no process that has to
outlive the session.

IT DOES NOT FOLLOW, AND THAT IS DELIBERATE. A ``--follow`` flag would put a
polling loop inside this command, and the tool that already does that job does
it better -- Monitor's own guidance is to write the loop in the shell so the
filter and the interval are visible at the call site. Following is
``fleet-watch ... ; sleep 5`` in a ``while``, or a ``tail -f`` on the feed
file, and both are the caller's to compose.

THE WEDGE DETECTOR LIVES HERE, because nothing else is looking. A dispatch
whose lease has expired with no terminal event cannot report its own death --
that is what being wedged means. :func:`lost_runs` finds those by comparing
the ledger's live rows against the leases that still exist, and reports them
as ``lost``. Without it a wedged run is indistinguishable from a slow one,
which is exactly how two suites held 77.9 GB for twenty-nine minutes on
2026-09-04 while looking like work in progress.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import get_logger, setup_logging

from fleet.cli import _config
from fleet.contracts.feed import FeedEvent, render_feed_line
from fleet.contracts.ledger import LedgerEntry, is_live
from fleet.core import _test_hooks, leases, records

_log = get_logger(__name__)

RUN_FLAG = "--run"

_FLAGS = (_config.CONFIG_FLAG, RUN_FLAG)


def lost_runs(loaded: _config.LoadedWorkspace, *, now_unix: int) -> tuple[LedgerEntry, ...]:
    """Find dispatches that are recorded as running but hold no live lease.

    A run holds its lease for as long as it runs, so a live ledger row with
    no lease behind it means the run stopped without saying so. That is the
    observable signature of a wedge, and it is observable only from outside.

    Args:
        loaded: The workspace and its resolved record paths.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        The rows that have been lost, in ledger order.
    """
    held = {lease["run_id"] for lease in leases.held_leases(loaded.leases, now_unix=now_unix)}
    return tuple(
        row
        for row in records.read_ledger(loaded.ledger)
        if is_live(row) and row["run_id"] not in held
    )


def lines_for(loaded: _config.LoadedWorkspace, *, run_id: str | None, now_unix: int) -> list[str]:
    """Render the feed, plus a line for every run found lost.

    Args:
        loaded: The workspace and its resolved record paths.
        run_id: Show only this dispatch's events, or None for all of them.
        now_unix: Current time, whole seconds since the epoch.

    Returns:
        One line per event, oldest first, with lost-run lines appended. Lost
        lines come last because they are derived rather than recorded: they
        describe the absence of an event, and interleaving them by timestamp
        would imply the run emitted something it did not.
    """
    rendered = [
        render_feed_line(event)
        for event in records.read_feed(loaded.feed)
        if run_id is None or event["run_id"] == run_id
    ]
    rendered.extend(
        _lost_line(row)
        for row in lost_runs(loaded, now_unix=now_unix)
        if run_id is None or row["run_id"] == run_id
    )
    return rendered


def _lost_line(row: LedgerEntry) -> str:
    """Render a lost dispatch in the feed's own shape.

    Built as a :class:`~fleet.contracts.feed.FeedEvent` and rendered by the
    feed's own renderer rather than formatted here, so a subscriber's filter
    matches it identically to a recorded event. A second spelling would be a
    line that greps differently from every other line for no reason a reader
    could see.

    Args:
        row: The ledger row that has no live lease.

    Returns:
        The rendered line.
    """
    return render_feed_line(
        FeedEvent(
            at_unix=row["started_unix"],
            run_id=row["run_id"],
            node=row["node"],
            project=row["project"],
            kind="lost",
            detail=(
                f"recorded running since {row['started_unix']} but holds no live lease; "
                f"dispatched by {row['agent']}"
            ),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print the event stream.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 always. A watcher reports what happened; whether what happened was
        a failure is on the lines, and a non-zero status here would make a
        subscriber's own shell treat reading the feed as an error.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing its value.
        AppError: If the feed or ledger holds a line that cannot be read.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    loaded = _config.load_workspace(parsed)

    for line in lines_for(loaded, run_id=parsed.get(RUN_FLAG), now_unix=_test_hooks.now()):
        _log.info("%s", line)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="fleet-watch",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["entrypoint", "lines_for", "lost_runs", "main"]


# Without this, `python -m fleet.cli.watch` imports the module, runs nothing
# and exits 0 -- an empty feed, which reads as "nothing is happening".
if __name__ == "__main__":
    entrypoint()
