"""Were the extractors never built, or built and destroyed?

The Hard rung splits perfectly on the final extractor count, but a final count
of zero has two opposite causes. "Never claimed a pool" is a race we lost;
"claimed and lost them" is ground we could not hold, and the fixes point in
different directions ([[policy-holding-ground]]).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path("C:/Users/Test/PROJECTS/API/clients/RustedWarfareBot")
TRACES = REPO / "runs" / "traces"
WON = {"31337", "4242", "555", "777"}


def _rows(path: Path) -> list[dict[str, int]]:
    """Read one trace as a list of column-name to value maps."""
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split()
    out: list[dict[str, int]] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) == len(header):
            out.append({name: int(value) for name, value in zip(header, fields)})
    return out


print(
    f"{'seed':>9}{'won':>5}{'peak':>6}{'end':>5}{'first@':>8}{'peak@':>8}"
    f"{'credits>=700':>14}{'max_credits':>12}"
)
for path in sorted(TRACES.glob("duel-s*.ndjson")):
    seed = path.stem.removeprefix("duel-s")
    rows = _rows(path)
    if not rows:
        continue
    span = rows[-1]["frame"] - rows[0]["frame"]
    peak = max(row["extractors"] for row in rows)
    first = next((row for row in rows if row["extractors"] >= 1), None)
    at_peak = next(row for row in rows if row["extractors"] == peak)
    rich = sum(1 for row in rows if row["credits"] >= 700)
    start = rows[0]["frame"]
    first_at = "--" if first is None else str((first["frame"] - start) * 100 // span) + "%"
    peak_at = str((at_peak["frame"] - start) * 100 // span) + "%"
    rich_pct = str(rich * 100 // len(rows)) + "%"
    won = "Y" if seed in WON else "."
    print(
        f"{seed:>9}{won:>5}{peak:>6}{rows[-1]['extractors']:>5}{first_at:>8}{peak_at:>8}"
        f"{rich_pct:>14}{max(row['credits'] for row in rows):>12}"
    )
