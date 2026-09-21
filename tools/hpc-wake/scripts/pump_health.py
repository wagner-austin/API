"""The pump's own health, written where something other than a log can read it.

THE OUTAGE THIS EXISTS FOR, 2026-09-16 to 2026-09-21. MCPs mig 514 made the
board refuse a write from a session no ledger surface knows. Every one of the
pump's three publishers is such a session, so all three began failing on every
tick the moment it deployed -- hpc-wake, ci-wake and lock-wake, roughly 2,700
ticks over five days and fourteen hours. Nothing said so. The scheduler
recorded a nonzero result nobody reads, the traceback went to ``runs/cycle.log``
which is truncated past 1 MB (it held 80 minutes of a five-day outage by the
time the outage was found), and every Claude session's start banner went on
telling it that it was subscribed to three feeds that had published nothing.

Seven red CI runs on 2026-09-21 reached nobody because of it, and the session
whose commit broke the build pushed three more times without ever being told.

WHAT THIS MODULE ADDS is a small, bounded, machine-readable record of what the
last tick did, per publisher, at a fixed path. It is NOT a log: it is one
record, rewritten whole each tick, that answers "is the wake system
publishing?" without reading anything else. The MCPs session-start hook reads
it and says so in the banner every session sees, which is the surface that was
lying.

TWO CONSTRAINTS SHAPE THE FORMAT, and neither is taste.

1. STDLIB ONLY, like its caller and for the same reason: the scheduled task
   runs these modules under the SYSTEM python, before any venv exists, so a
   first-party import resolves against whatever stale site-packages that
   interpreter happens to hold, or nothing (measured 2026-09-09, when every
   scheduled cycle exited 1 before writing a log header).
2. This monorepo's guard bans the stdlib ``json`` module outright: JSON is
   read and written through ``platform_core.json_utils``, which constraint 1
   forbids here. So the record is LINE-ORIENTED -- tab-separated, one record
   per line, parsed with ``str.split`` -- which needs no parser at all and is
   readable by the hook, by ``Get-Content`` and by a person.
"""

from __future__ import annotations

import datetime
import pathlib
from collections.abc import Sequence

from typing_extensions import TypedDict

#: The record's filename, beside ``cycle.log`` in the pump's ``runs/``.
HEALTH_FILENAME = "pump-health.tsv"

#: The record's shape version, so a reader can refuse a shape it predates
#: rather than mis-read one field as another.
HEALTH_VERSION = 1

#: The line kind carrying one publisher's outcome.
PUBLISHER_KEY = "publisher"

#: The line kind carrying the tick's own clock.
WRITTEN_KEY = "written"

#: The line kind carrying :data:`HEALTH_VERSION`.
VERSION_KEY = "version"

#: What a publisher that has never been seen to succeed records as its last
#: success. The empty string rather than an epoch, which would read as a
#: success in 1970 and sort like one.
NEVER = ""


class PublisherHealth(TypedDict):
    """One publisher's outcome, as of the tick that wrote this record.

    Attributes:
        name: The publisher's marker, matching the cycle log's ``-- <name>``.
        exit_code: The process's exit status on this tick. 0 is published.
        consecutive_failures: How many ticks in a row have ended nonzero,
            including this one. The number, not a flag, because "failing
            since the last tick" and "failing for five days" are different
            findings and only one of them is an outage.
        last_ok: When this publisher last exited 0, as an ISO-8601 UTC
            instant, or :data:`NEVER`. An instant rather than an age: an age
            computed at write time is wrong by however long the reader took
            to arrive.
    """

    name: str
    exit_code: int
    consecutive_failures: int
    last_ok: str


class PumpHealth(TypedDict):
    """The whole record.

    Attributes:
        version: :data:`HEALTH_VERSION`.
        written: When this tick wrote the record, ISO-8601 UTC. A reader
            compares it against the pump's own interval to tell a failing
            publisher from a pump that has stopped ticking at all, which
            are different repairs.
        publishers: One entry per publisher, in publication order.
    """

    version: int
    written: str
    publishers: list[PublisherHealth]


def _as_int(text: str) -> int | None:
    """Read an exit code or a count, without trusting the file.

    Args:
        text: One field.

    Returns:
        The number, or ``None`` when the field is not one. Negative is
        accepted: a POSIX signal death is reported as a negative code, and
        a reader that dropped those would call a killed publisher unknown.
    """
    if text.startswith("-"):
        return -int(text[1:]) if text[1:].isdigit() else None
    return int(text) if text.isdigit() else None


def _decode_publisher(fields: Sequence[str]) -> PublisherHealth | None:
    """Read one publisher line's fields.

    Args:
        fields: The tab-separated fields AFTER the leading line kind.

    Returns:
        The entry, or ``None`` when the line is not the shape this module
        writes. A previous record is HISTORY, not input the pump depends on:
        an unreadable line costs one publisher's failure streak its memory
        and nothing else, so it is dropped rather than raised on, and the
        tick still publishes. A tick that refused to run because its own
        bookkeeping was malformed would be this outage with a new cause.
    """
    if len(fields) != 4:
        return None
    exit_code = _as_int(fields[1])
    failures = _as_int(fields[2])
    if fields[0] == "" or exit_code is None or failures is None:
        return None
    return PublisherHealth(
        name=fields[0],
        exit_code=exit_code,
        consecutive_failures=failures,
        last_ok=fields[3],
    )


def read_health(path: pathlib.Path) -> dict[str, PublisherHealth]:
    """Read the previous tick's record, by publisher name.

    Args:
        path: The record's path.

    Returns:
        The previous entries, or an empty mapping when there is no readable
        record. The FIRST tick after this module ships has none, and so does
        the tick after a machine is rebuilt.
    """
    if not path.exists():
        return {}
    found: dict[str, PublisherHealth] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields[0] != PUBLISHER_KEY:
            continue
        entry = _decode_publisher(fields[1:])
        if entry is not None:
            found[entry["name"]] = entry
    return found


def next_health(
    previous: dict[str, PublisherHealth],
    outcomes: Sequence[tuple[str, int]],
    now: datetime.datetime,
) -> PumpHealth:
    """Fold this tick's exit codes into the record the next one will read.

    Args:
        previous: Output of :func:`read_health`.
        outcomes: ``(publisher name, exit code)`` in publication order.
        now: This tick's clock, timezone-aware UTC.

    Returns:
        The record to write.
    """
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    publishers: list[PublisherHealth] = []
    for name, exit_code in outcomes:
        before = previous.get(name)
        if exit_code == 0:
            publishers.append(
                PublisherHealth(name=name, exit_code=0, consecutive_failures=0, last_ok=stamp)
            )
            continue
        publishers.append(
            PublisherHealth(
                name=name,
                exit_code=exit_code,
                consecutive_failures=(0 if before is None else before["consecutive_failures"]) + 1,
                last_ok=NEVER if before is None else before["last_ok"],
            )
        )
    return PumpHealth(version=HEALTH_VERSION, written=stamp, publishers=publishers)


def render_health(health: PumpHealth) -> str:
    """Render the record as the lines :func:`read_health` reads back.

    Args:
        health: The record.

    Returns:
        The file's whole text, newline-terminated.
    """
    lines = [
        f"{VERSION_KEY}\t{health['version']}",
        f"{WRITTEN_KEY}\t{health['written']}",
    ]
    lines.extend(
        "\t".join(
            [
                PUBLISHER_KEY,
                entry["name"],
                str(entry["exit_code"]),
                str(entry["consecutive_failures"]),
                entry["last_ok"],
            ]
        )
        for entry in health["publishers"]
    )
    return "\n".join(lines) + "\n"


def write_health(path: pathlib.Path, health: PumpHealth) -> None:
    """Replace the record.

    Written whole rather than appended to, and that is the point: a log
    grows until it is truncated and then forgets the outage it recorded,
    which is exactly what happened to ``cycle.log`` between 2026-09-16 and
    2026-09-21. One record, always current, bounded by the publisher count.

    Args:
        path: The record's path.
        health: The record.
    """
    path.write_text(render_health(health), encoding="utf-8")
