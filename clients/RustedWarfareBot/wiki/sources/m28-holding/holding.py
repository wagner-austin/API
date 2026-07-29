"""When do the extractors die, and what is the army doing when they do?

Every seed reaches three extractors. Winners keep two or three; the rest end on
nought or one. So the question is not how to claim a pool -- it is what happens
between the peak and the end.
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


print(f"{'seed':>9}{'won':>5}{'drops':>7}{'regained':>10}{'army@drop':>11}{'army_avg':>10}")
for path in sorted(TRACES.glob("duel-s*.ndjson")):
    seed = path.stem.removeprefix("duel-s")
    rows = _rows(path)
    if not rows:
        continue
    drops = 0
    regains = 0
    army_at_drop: list[int] = []
    for before, after in zip(rows, rows[1:]):
        if after["extractors"] < before["extractors"]:
            drops += before["extractors"] - after["extractors"]
            army_at_drop.append(before["army"])
        if after["extractors"] > before["extractors"]:
            regains += after["extractors"] - before["extractors"]
    avg_drop = sum(army_at_drop) // len(army_at_drop) if army_at_drop else 0
    avg_army = sum(row["army"] for row in rows) // len(rows)
    won = "Y" if seed in WON else "."
    print(f"{seed:>9}{won:>5}{drops:>7}{regains:>10}{avg_drop:>11}{avg_army:>10}")
