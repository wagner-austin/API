"""The fleet drawn as one page: every lane, every batch, every verdict.

The door already speaks HTTP and the queue already mirrors every
scorecard's verdict onto its row, so a dashboard costs one route and one
renderer -- no framework, no JavaScript, no second data path. The page is
rendered server-side from the same reads the NDJSON surface uses and
refreshes itself with a plain meta tag; what a browser shows is exactly
what :func:`rw_bot.service.queue.batch_results` reports, drawn as tables.

One glance answers the fleet's three standing questions: what each engine
lane is playing right now, what is waiting behind it, and how each
experiment's arms are scoring against their controls
([[harness-match-service]]). The look is the Tankpit fleet page's glass
idiom, kept server-side because match verdicts change by the minute, not
the second.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from rw_bot.service._test_hooks import Connection
from rw_bot.service.queue import (
    JobResult,
    RunningMatch,
    batch_results,
    fleet_batches,
    running_matches,
)

#: Verdict words a scorecard can carry, in column order: the two upper
#: outcomes first, then the two ways a match is lost.
VERDICTS: tuple[str, ...] = ("won", "survived", "defeated", "wiped")

#: Every column after the arm name, in display order.
_COLUMNS: tuple[str, ...] = ("queued", "running", "done", *VERDICTS, "failed")

#: The static page head: dark glass, neon header, self-refreshing --
#: the Tankpit fleet page's visual language. Styling lives here so the
#: renderer emits structure only.
_HEAD = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<meta http-equiv='refresh' content='10'>"
    "<title>match fleet</title><style>"
    "body{margin:0;color:#c9d1d9;font-family:Consolas,monospace;"
    "background:#0c0f16 radial-gradient(1100px 700px at 18% -8%,"
    "rgba(37,52,120,.55),transparent 62%) no-repeat fixed;"
    "padding:1.3rem 1.6rem}"
    "h1{margin:0 0 .2rem;font-size:1.2rem;letter-spacing:.06em;"
    "color:rgb(57,255,20);text-shadow:0 0 8px rgba(57,255,20,.45)}"
    "h2{color:#79c0ff;font-size:1rem;margin:1.4rem 0 .4rem;"
    "letter-spacing:.05em}"
    "p.sub{color:#9aa3b5;margin:.2rem 0 1rem}"
    "table{border-collapse:collapse;background:rgba(24,34,80,.28);"
    "border-radius:8px;box-shadow:0 8px 24px rgba(0,0,0,.6)}"
    "td,th{border-bottom:1px solid rgba(255,255,255,.07);"
    "padding:.4rem .8rem;text-align:right;white-space:nowrap}"
    "th{color:#9aa3b5;font-size:.76rem;text-transform:uppercase;"
    "letter-spacing:.07em}"
    "td.arm,th.arm{text-align:left;color:#c9d1d9}"
    ".won{color:#3fb950;font-weight:600}.survived{color:#58a6ff}"
    ".defeated{color:#d29922}.wiped{color:#f85149}.failed{color:#f85149}"
    ".running{color:#e3b341}.queued{color:#9aa3b5}.done{color:#c9d1d9}"
    ".muted{color:#484f58}"
    "</style></head><body>"
)


def render_dashboard(conn: Connection) -> str:
    """Draw the whole queue as one self-refreshing HTML page.

    Args:
        conn: An open queue connection; every read rolls back, exactly as
            the reads it composes do.

    Returns:
        The complete page: the fleet summary, the busy lanes, then one
        table per batch, newest batch first.

    Raises:
        MatchServiceError: ``RW-SERVICE-001`` when any underlying row is
            unreadable.
    """
    lanes = running_matches(conn)
    batches = fleet_batches(conn)
    queued = 0
    sections: list[str] = []
    for batch in batches:
        results = batch_results(conn, batch)
        queued += sum(1 for result in results if result["state"] == "queued")
        sections.append(_batch_section(batch, results))
    summary = (
        f"<p class='sub'>{len(batches)} batches &middot;"
        f" <span class='running'>{len(lanes)} running</span> &middot;"
        f" <span class='queued'>{queued} queued</span></p>"
    )
    return (
        _HEAD
        + "<h1>match fleet</h1>"
        + summary
        + _lanes_section(lanes)
        + "".join(sections)
        + "</body></html>"
    )


def _lanes_section(lanes: tuple[RunningMatch, ...]) -> str:
    """Draw what every busy engine lane is playing.

    Args:
        lanes: The running matches, ordered by clone index.

    Returns:
        The lanes table, or an idle note when nothing runs.
    """
    if not lanes:
        return "<h2>lanes</h2><p class='sub'>all lanes idle</p>"
    header = (
        "<h2>lanes</h2><table><tr><th>lane</th><th class='arm'>batch</th>"
        "<th class='arm'>arm</th><th>seed</th><th class='arm'>worker</th></tr>"
    )
    rows = "".join(
        f"<tr><td class='running'>{lane['clone_index']}</td>"
        f"<td class='arm'>{escape(lane['batch'])}</td>"
        f"<td class='arm'>{escape(lane['label'])}</td>"
        f"<td>{lane['seed']}</td>"
        f"<td class='arm'>{escape(lane['worker'])}</td></tr>"
        for lane in lanes
    )
    return header + rows + "</table>"


def _batch_section(batch: str, results: tuple[JobResult, ...]) -> str:
    """Draw one batch as a table with one row per arm.

    Args:
        batch: The batch name, escaped into the heading.
        results: The batch's per-job outcomes.

    Returns:
        The heading and table markup.
    """
    arms: dict[str, dict[str, int]] = {}
    for result in results:
        tally = arms.setdefault(result["label"], dict.fromkeys(_COLUMNS, 0))
        if result["state"] in tally:
            tally[result["state"]] += 1
        if result["verdict"] in VERDICTS:
            tally[result["verdict"]] += 1
    heading = f"<h2>{escape(batch)}</h2>"
    header = (
        "<table><tr><th class='arm'>arm</th>"
        + "".join(f"<th>{name}</th>" for name in _COLUMNS)
        + "</tr>"
    )
    rows = "".join(_arm_row(label, arms[label]) for label in sorted(arms))
    return heading + header + rows + "</table>"


def _arm_row(label: str, tally: Mapping[str, int]) -> str:
    """Draw one arm's row, muting the zeroes so the live numbers pop.

    Args:
        label: The arm's label.
        tally: Counts per column name.

    Returns:
        The row markup.
    """
    cells = "".join(
        f"<td class='{name if tally[name] else 'muted'}'>{tally[name]}</td>" for name in _COLUMNS
    )
    return f"<tr><td class='arm'>{escape(label)}</td>{cells}</tr>"


__all__ = ["VERDICTS", "render_dashboard"]
