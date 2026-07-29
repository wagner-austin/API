"""Are the extractors dying at the front, or being picked off at home?

A loss is "left the roster", which an upgrade also does -- so the first thing
this checks is whether the winners, whose income proves they upgraded, record
any extractor departure at all. If they do not, the losers' entries are deaths.

Then: where. The army's own losses mark where the fighting is, so an extractor
dying inside that cloud is ground lost at the front and one dying far outside it
was raided while the army was elsewhere.
"""

from __future__ import annotations

from math import hypot
from pathlib import Path

REPO = Path("C:/Users/Test/PROJECTS/API/clients/RustedWarfareBot")
TRACES = REPO / "runs" / "traces"
WON = {"31337", "4242", "555", "777"}
EXTRACTORS = ("extractorT1", "extractorT2", "extractorT3")


def _losses(path: Path) -> list[tuple[int, str, float, float]]:
    """Read the per-loss table: frame, type, x, y."""
    out: list[tuple[int, str, float, float]] = []
    seen = False
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:2] == ["frame", "unit"]:
            seen = True
            continue
        if not seen or len(fields) != 5:
            continue
        out.append((int(fields[0]), fields[2], float(fields[3]), float(fields[4])))
    return out


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """The middle of a cloud of positions."""
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


print(f"{'seed':>9}{'won':>5}{'extractor deaths':>18}{'at % of run':>13}   distance from the army's front")
for path in sorted(TRACES.glob("duel-s*.ndjson")):
    seed = path.stem.removeprefix("duel-s")
    losses = _losses(path)
    if not losses:
        continue
    span = max(frame for frame, _, _, _ in losses) or 1
    tanks = [(x, y) for _, name, x, y in losses if name == "c_tank"]
    gone = [(f, x, y) for f, name, x, y in losses if name in EXTRACTORS]
    won = "Y" if seed in WON else "."
    if not gone:
        print(f"{seed:>9}{won:>5}{0:>18}{'--':>13}   --")
        continue
    front = _centroid(tanks) if tanks else (0.0, 0.0)
    spread = (
        sum(hypot(x - front[0], y - front[1]) for x, y in tanks) / len(tanks) if tanks else 0.0
    )
    when = ", ".join(str(f * 100 // span) + "%" for f, _, _ in gone)
    away = ", ".join(str(round(hypot(x - front[0], y - front[1]))) for _, x, y in gone)
    print(f"{seed:>9}{won:>5}{len(gone):>18}{when:>13}   {away}  (army spread {spread:.0f})")
