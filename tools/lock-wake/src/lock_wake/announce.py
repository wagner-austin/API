"""Folding journal transitions into one board post per cycle.

THE NOISE BUDGET IS A DESIGN INPUT, NOT AN AFTERTHOUGHT (board 9406cfd9,
acceptance 6, operator-stated): publishers batch per tick, always. One
cycle produces AT MOST ONE post, covering every hold that crossed a
BOUNDARY in the slice -- acquired (a cascade started), released (it
finished), failed, timeout, and the two pre-hold refusals gate-blocked
and refused (a session told "no" before any hold existed). The
journal's progress kinds -- requested,
waiting, step -- are folded into their hold's line as counts when a
boundary is present, and consumed silently when not: a 30-minute deploy
writes step lines every tick, and a bridge that posted each tick would be
the spam this system exists to prevent. The spec's own words: publish
state CHANGES, "a cascade started/ended".

COUNTS ARE PER SLICE, STATED AS SUCH. A hold's steps may arrive across
several ticks; each boundary line reports the counts seen in ITS window
("+3 step(s) this window"), which is honest and cheap, where a
cross-tick tally would need a second position record to say something
nobody acts on.

CHECK RUNS RIDE THE SAME POST, UNADDRESSED (MCPs board task ea2ea29c).
A ``checked`` row is one finished ``make test`` run, and every such row in
the slice becomes one line under a CHECKS heading in the same single post.
Those lines name the runner in plain text and never ``@`` it: the runner
already watched its own output, and a mention would buy it a wake or a
turn-boundary continuation for every run it made. A session that wants
check results subscribes to this bridge's standing task, and the
subscription is what makes the post reach it.

ACTIONS RIDE IT TOO, AND ONLY WHEN THEY DID NOT FINISH (MCPs board task
b00ab472). Every MCPs command that changes state and takes no lock (a
backup, a restore, a production Keycloak account, a mirror push) journals
its start, each line it said and how it ended into ``.action-events.jsonl``
beside the fleet journal. An action is not a hold, so its rows are passed
apart and posted under their own ACTIONS heading; one that ended ``failed``
or ``refused`` gets a line, and its session an ``@``, because that is the
result nobody watching a terminal saw. A clean finish is a ``step`` and
stays silent, as a hold's progress does: a mirror push runs on every commit
of fifteen repositories.
"""

from __future__ import annotations

import datetime
from typing import Final

from typing_extensions import TypedDict

from lock_wake.journal import LockEvent

#: Kinds that make a hold worth a line in the post. ``gate-blocked`` and
#: ``refused`` are single-event stories -- a freshness gate or the lock
#: wrapper telling a session no, with no hold ever existing -- and each is a
#: boundary because a refusal nobody hears repeats itself: four redundant
#: rebuilds ran in 20 minutes on 2026-09-11 precisely because being told
#: "no" (or "wait") left no record anyone else could see (MCPs 07fcc6af).
BOUNDARY_KINDS: Final = ("acquired", "released", "failed", "timeout", "gate-blocked", "refused")

#: Kinds that end a hold's line: the terminal outcomes plus the two
#: pre-hold refusals, which are their own beginning and end.
_ENDING_KINDS: Final = ("released", "failed", "timeout", "gate-blocked", "refused")

#: Kinds that make an action worth a line: it did not finish. A finished
#: action closes with a ``step``, which is silent.
ACTION_BOUNDARY_KINDS: Final = ("failed", "refused")


class Announcement(TypedDict):
    """One cycle's post, ready for the board.

    Attributes:
        body: The full post text.
        agents: Distinct non-empty session labels behind the announced
            holds and actions, in first-seen order -- the @mention targets.
            Check runs add none.
        holds: How many holds the post covers, for the cycle's report line.
        checks: How many finished check runs the post covers.
        actions: How many actions that did not finish the post covers.
    """

    body: str
    agents: tuple[str, ...]
    holds: int
    checks: int
    actions: int


def parse_journal_ts(ts: str) -> datetime.datetime:
    """Parse the journal's timestamp form.

    The lock wrapper writes seven fractional digits
    (``yyyy-MM-ddTHH:mm:ss.fffffffZ``); Python's microseconds stop at six,
    so the seventh is dropped -- a stated truncation of 100-nanosecond
    precision nothing here needs, not a lenient parse: everything else
    about the form is required exactly.

    Args:
        ts: The journal row's ``ts`` field.

    Returns:
        The timestamp, UTC.

    Raises:
        ValueError: A timestamp not in the journal's form.
    """
    if not ts.endswith("Z"):
        raise ValueError(f"journal timestamp {ts!r} does not end in Z")
    head, dot, fraction = ts[:-1].partition(".")
    trimmed = f"{head}{dot}{fraction[:6]}" if dot == "." else head
    return datetime.datetime.fromisoformat(trimmed).replace(tzinfo=datetime.UTC)


def _hold_line(events: tuple[LockEvent, ...]) -> str:
    """Compose one hold's line from its slice events.

    Args:
        events: Every event one pid wrote in this slice, in file order.

    Returns:
        The line.
    """
    first = events[0]
    kinds = [event["kind"] for event in events]
    steps = kinds.count("step")
    waits = kinds.count("waiting")
    parts: list[str] = [f"{first['label']} ({first['op']}, pid {first['holder_pid']}):"]
    if "acquired" in kinds:
        acquired_at = next(e for e in events if e["kind"] == "acquired")
        parts.append(f"acquired {acquired_at['ts'][11:19]}Z")
    ended = next((e for e in events if e["kind"] in _ENDING_KINDS), None)
    if ended is not None:
        if "acquired" in kinds:
            acquired_at = next(e for e in events if e["kind"] == "acquired")
            seconds = int(
                (
                    parse_journal_ts(ended["ts"]) - parse_journal_ts(acquired_at["ts"])
                ).total_seconds()
            )
            parts.append(f"{ended['kind'].upper()} after {seconds}s")
        else:
            parts.append(f"{ended['kind'].upper()} {ended['ts'][11:19]}Z")
        if ended["detail"] != "":
            parts.append(f"({ended['detail']})")
    if steps > 0:
        parts.append(f"+{steps} step(s) this window")
    if waits > 0:
        parts.append(f"+{waits} wait(s)")
    if first["agent"] != "":
        parts.append(f"by @{first['agent']}")
    return " ".join(parts)


def _check_line(event: LockEvent) -> str:
    """Compose one finished check run's line.

    Args:
        event: The ``checked`` row.

    Returns:
        The line, naming the runner without an ``@`` (module docstring).
    """
    runner = "" if event["agent"] == "" else f" by {event['agent']}"
    return f"{event['label']} {event['ts'][11:19]}Z: {event['detail']}{runner}"


def _by_process(
    events: tuple[LockEvent, ...], boundaries: tuple[str, ...]
) -> dict[int, tuple[LockEvent, ...]]:
    """Group a slice's rows by the process that wrote them, keeping those that crossed a boundary.

    Args:
        events: Rows of one kind of record, in file order.
        boundaries: The kinds that make a process's rows worth a line.

    Returns:
        Each such process's rows, in first-seen order.
    """
    grouped: dict[int, list[LockEvent]] = {}
    for event in events:
        grouped.setdefault(event["holder_pid"], []).append(event)
    return {
        holder: tuple(rows)
        for holder, rows in grouped.items()
        if any(event["kind"] in boundaries for event in rows)
    }


def _mention(agents: list[str], what: str) -> str | None:
    """The line that tags each session behind a group of lines, or None for nobody.

    Args:
        agents: The labels, in first-seen order, empty ones excluded.
        what: What happened to their operations, ending the line.

    Returns:
        The line, or None when no label is known.
    """
    if len(agents) == 0:
        return None
    return " ".join(f"@{agent}" for agent in agents) + f" {what}"


def _labels(groups: dict[int, tuple[LockEvent, ...]]) -> list[str]:
    """The distinct non-empty labels behind groups of rows, in first-seen order.

    Args:
        groups: Each process's rows.

    Returns:
        The labels.
    """
    labels: list[str] = []
    for rows in groups.values():
        agent = rows[0]["agent"]
        if agent != "" and agent not in labels:
            labels.append(agent)
    return labels


def announcement(
    events: tuple[LockEvent, ...], actions: tuple[LockEvent, ...], host: str
) -> Announcement | None:
    """Fold one slice's rows into one post, or nothing worth posting.

    Args:
        events: The slice's fleet-lock and check rows, in file order.
        actions: The slice's action rows (module docstring), in file order.
        host: The machine whose journals the rows came from (``hub``,
            ``diphtheria``), named in each heading and mention line: a pid
            and a label mean nothing until the reader knows which machine
            they ran on (MCPs board task 03590bf9).

    Returns:
        The post, or None when no hold crossed a boundary, no check run
        finished and every action finished -- progress-only slices are
        consumed silently by design (module docstring).
    """
    checks = tuple(event for event in events if event["kind"] == "checked")
    holds = _by_process(
        tuple(event for event in events if event["kind"] != "checked"), BOUNDARY_KINDS
    )
    unfinished = _by_process(actions, ACTION_BOUNDARY_KINDS)
    if len(holds) == 0 and len(checks) == 0 and len(unfinished) == 0:
        return None
    lines: list[str] = []
    if len(holds) > 0:
        lines.append(f"FLEET-LOCK on {host}: {len(holds)} hold(s) transitioned")
        lines.extend(_hold_line(hold) for hold in holds.values())
    if len(unfinished) > 0:
        lines.append(f"ACTIONS on {host}: {len(unfinished)} action(s) did not finish")
        lines.extend(_hold_line(action) for action in unfinished.values())
    if len(checks) > 0:
        lines.append(f"CHECKS: {len(checks)} make test run(s) finished")
        lines.extend(_check_line(event) for event in checks)
    holders = _labels(holds)
    actors = _labels(unfinished)
    for mention in (
        _mention(holders, f"your fleet-lock operation on {host} transitioned"),
        _mention(actors, f"your action on {host} did not finish"),
    ):
        if mention is not None:
            lines.append(mention)
    agents = [*holders, *(actor for actor in actors if actor not in holders)]
    return Announcement(
        body="\n".join(lines),
        agents=tuple(agents),
        holds=len(holds),
        checks=len(checks),
        actions=len(unfinished),
    )


__all__ = [
    "ACTION_BOUNDARY_KINDS",
    "BOUNDARY_KINDS",
    "Announcement",
    "announcement",
    "parse_journal_ts",
]
