"""What does the bot lose in a duel, and where?

The per-loss table answers "where", which is what separates dying at the enemy
front from being picked off at home. The 1v4 reading of it -- nothing dies at
the base -- is what redirected defence to the extractors; this asks the same
question of the duels, where the extractor losses actually decide the match.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path("C:/Users/Test/PROJECTS/API/clients/RustedWarfareBot")
TRACES = REPO / "runs" / "traces"
WON = {"31337", "4242", "555", "777"}


def _losses(path: Path) -> list[tuple[int, str, float, float]]:
    """Read the per-loss table: frame, type, x, y."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[tuple[int, str, float, float]] = []
    seen_second_header = False
    for line in lines:
        fields = line.split()
        if fields[:2] == ["frame", "unit"]:
            seen_second_header = True
            continue
        if not seen_second_header or len(fields) != 5:
            continue
        out.append((int(fields[0]), fields[1 + 1], float(fields[3]), float(fields[4])))
    return out


print(f"{'seed':>9}{'won':>5}{'losses':>8}   what it loses, commonest first")
totals: dict[str, int] = {}
for path in sorted(TRACES.glob("duel-s*.ndjson")):
    seed = path.stem.removeprefix("duel-s")
    losses = _losses(path)
    counted: dict[str, int] = {}
    for _, type_name, _x, _y in losses:
        counted[type_name] = counted.get(type_name, 0) + 1
        totals[type_name] = totals.get(type_name, 0) + 1
    mix = ", ".join(
        f"{name} x{n}" for name, n in sorted(counted.items(), key=lambda p: (-p[1], p[0]))[:5]
    )
    won = "Y" if seed in WON else "."
    print(f"{seed:>9}{won:>5}{len(losses):>8}   {mix}")

print("\nacross all twelve:")
for name, n in sorted(totals.items(), key=lambda p: (-p[1], p[0])):
    print(f"  {name:<20}{n:>6}")
