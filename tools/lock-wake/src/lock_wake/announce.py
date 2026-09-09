"""Folding journal transitions into one board post per cycle.

THE NOISE BUDGET IS A DESIGN INPUT, NOT AN AFTERTHOUGHT (board 9406cfd9,
acceptance 6, operator-stated): publishers batch per tick, always. One
cycle produces AT MOST ONE post, covering every hold that crossed a
BOUNDARY in the slice -- acquired (a cascade started), released (it
finished), failed, timeout. The journal's progress kinds -- requested,
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
"""

from __future__ import annotations

import datetime
from typing import Final

from typing_extensions import TypedDict

from lock_wake.journal import LockEvent

#: Kinds that make a hold worth a line in the post.
BOUNDARY_KINDS: Final = ("acquired", "released", "failed", "timeout")


class Announcement(TypedDict):
    """One cycle's post, ready for the board.

    Attributes:
        body: The full post text.
        agents: Distinct non-empty session labels behind the announced
            holds, in first-seen order -- the @mention targets.
        holds: How many holds the post covers, for the cycle's report line.
    """

    body: str
    agents: tuple[str, ...]
    holds: int


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
    ended = next((e for e in events if e["kind"] in ("released", "failed", "timeout")), None)
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


def announcement(events: tuple[LockEvent, ...]) -> Announcement | None:
    """Fold one slice's events into one post, or nothing worth posting.

    Args:
        events: The slice's events, in file order.

    Returns:
        The post, or None when no hold crossed a boundary -- progress-only
        slices are consumed silently by design (module docstring).
    """
    holds: dict[int, list[LockEvent]] = {}
    for event in events:
        holds.setdefault(event["holder_pid"], []).append(event)
    announced = {
        holder: tuple(hold)
        for holder, hold in holds.items()
        if any(event["kind"] in BOUNDARY_KINDS for event in hold)
    }
    if len(announced) == 0:
        return None
    lines = [f"FLEET-LOCK: {len(announced)} hold(s) transitioned"]
    agents: list[str] = []
    for hold_events in announced.values():
        lines.append(_hold_line(hold_events))
        agent = hold_events[0]["agent"]
        if agent != "" and agent not in agents:
            agents.append(agent)
    if len(agents) > 0:
        mentions = " ".join(f"@{agent}" for agent in agents)
        lines.append(f"{mentions} your fleet-lock operation transitioned")
    return Announcement(body="\n".join(lines), agents=tuple(agents), holds=len(announced))


__all__ = ["BOUNDARY_KINDS", "Announcement", "announcement", "parse_journal_ts"]
