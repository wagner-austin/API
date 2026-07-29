"""Tabulate any two sweep batches against each other.

Written because the same three questions were being re-asked by hand of every
batch, and asking them by hand is how the extractor-defence arm got written up
as a fix before its own numbers were read.

The three that matter, and why each is here rather than the figures a scorecard
makes prominent:

* **verdicts** -- the only thing the goal is stated in.
* **extractor drops**, from the per-sample trace. Holding a pool decides the
  duel; the ending count cannot tell "never built" from "built and destroyed"
  ([[policy-holding-ground]]).
* **engageable share**, from the report. A gap between what is visible and what
  the army can shoot is the air problem, and it is the only figure that says
  whether a composition change did the thing it was chosen to do
  ([[mechanics-combat-profile]]).

Income and total worth are deliberately absent. The change that took Easy from
3/12 to 12/12 lowered both, so a comparison led by them would have rejected it.

Run as ``python compare.py <batch-a> <batch-b> ...`` from the repository root.

**A trace belongs to whichever run wrote it last, and the drops column has to
prove it.** ``runs/traces/duel-s<seed>.ndjson`` is overwritten by every batch,
so an older batch listed beside a newer one would silently borrow the newer
one's traces and report drops that belong to the other arm -- the precise class
of quietly-wrong figure this file exists to stop.

Two checks, because the first one alone is not enough and the way it failed is
worth recording. Matching the trace's row count against the report's ``samples
seen`` looks sufficient and is not: **every match that runs out the clock has
exactly 4,000 rows**, so three seeds that hit the limit in both arms passed the
check while holding the other arm's trace. The row count identifies a length,
not a run. So the write times must agree as well -- a match writes its trace and
its report within seconds of each other, and two batches are minutes apart at
best.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("runs/sweeps")
TRACES = Path("runs/traces")

VERDICT = re.compile(r"^verdict\s+(\w+)")
SEEN = re.compile(r"^enemies seen\s+\d+ -> (\d+) \((\d+) engageable\)")
SAMPLES = re.compile(r"^samples seen\s+(\d+)")
WORTH = re.compile(r"^total worth\s+\d+ -> (\d+)")
RIVAL = re.compile(r"^best rival\s+\d+ -> (\d+)")


#: Seconds a trace and its own report may differ by in write time.
#:
#: One match writes both as it finishes. Two batches are minutes apart at the
#: very least, so this only has to be tighter than that.
_SAME_RUN_SECONDS = 120


def _drops(seed: str, samples: int, report: Path) -> str:
    """Net extractor losses, or blank when the trace is another run's.

    Args:
        seed: The match seed, which names the trace.
        samples: Observations the report says this match saw. The trace must
            hold exactly that many rows.
        report: The result file, whose write time the trace's must match.

    Returns:
        The drop count as text, or empty when there is no trace or it belongs
        to a different run.
    """
    path = TRACES / f"duel-s{seed}.ndjson"
    if not path.exists():
        return ""
    if abs(path.stat().st_mtime - report.stat().st_mtime) > _SAME_RUN_SECONDS:
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split()
    counts = [
        int(dict(zip(header, fields))["extractors"])
        for fields in (line.split() for line in lines[1:])
        if len(fields) == len(header)
    ]
    if len(counts) != samples:
        return ""
    return str(sum(max(0, a - b) for a, b in zip(counts, counts[1:])))


def _report(batch: str) -> None:
    """Print one batch's table and its verdict tally."""
    folder = ROOT / batch
    if not folder.exists():
        print(f"=== {batch}: no such batch")
        return
    print(f"=== {batch}")
    print(
        f"{'seed':>9}{'verdict':>12}{'samples':>9}{'drops':>7}"
        f"{'visible':>9}{'engageable':>12}{'%':>6}{'margin':>8}"
    )
    tally: dict[str, int] = {}
    wins: list[int] = []
    margins: list[float] = []
    routs: list[str] = []
    for path in sorted(folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8").splitlines()

        def first(pattern: re.Pattern[str], lines: list[str] = text) -> re.Match[str] | None:
            return next((m for m in (pattern.match(line) for line in lines) if m), None)

        verdict = first(VERDICT)
        seen = first(SEEN)
        samples = first(SAMPLES)
        name = verdict.group(1) if verdict else "?"
        tally[name] = tally.get(name, 0) + 1
        visible, engaged = (int(seen.group(1)), int(seen.group(2))) if seen else (0, 0)
        seed = path.stem.removeprefix("duel-s")
        if name == "won" and samples:
            wins.append(int(samples.group(1)))
        # How crushing, not merely whether. Endpoint worth against the strongest
        # opponent's: a win at 1.1x and a win at 12x are the same row in a
        # verdict column and very different games ([[policy-verdict]]).
        ours = first(WORTH)
        theirs = first(RIVAL)
        margin = ""
        if ours and theirs:
            mine, rival = int(ours.group(1)), int(theirs.group(1))
            if not rival:
                # **A rival worth of nothing is the most crushing result there
                # is, and dividing by it rendered a blank.** Left as a gap these
                # read as missing measurements and were silently dropped from
                # the median -- so the best games were the ones the summary
                # threw away.
                margin = "total" if mine else "--"
                if mine:
                    routs.append(seed)
            else:
                margins.append(mine / rival)
                margin = f"{mine / rival:.1f}x"
        print(
            f"{seed:>9}{name:>12}{(samples.group(1) if samples else '?'):>9}"
            f"{_drops(seed, int(samples.group(1)) if samples else -1, path):>7}"
            f"{visible:>9}{engaged:>12}"
            f"{(str(engaged * 100 // visible) + '%' if visible else '--'):>6}{margin:>8}"
        )
    won = tally.get("won", 0)
    lost = tally.get("defeated", 0) + tally.get("wiped", 0)
    print(f"  {won} won, {lost} lost, {tally}")
    if wins:
        wins.sort()
        middle = wins[len(wins) // 2]
        print(f"  samples to win: fastest {wins[0]}, median {middle}, slowest {wins[-1]}")
    if routs:
        print(f"  opponent left with nothing at all: {len(routs)} ({', '.join(routs)})")
    if margins:
        margins.sort()
        print(
            f"  margin where anything survived (our worth / theirs): worst {margins[0]:.1f}x, "
            f"median {margins[len(margins) // 2]:.1f}x, best {margins[-1]:.1f}x"
        )
    print()


for batch_name in sys.argv[1:] or ["duel-hard", "duel-hard-aa"]:
    _report(batch_name)
