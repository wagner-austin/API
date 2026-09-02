"""Compare two batches seed by seed, which is where the resolution lives.

Comparing panel totals throws away the pairing: 13W against 14W reads as
noise, while the seed-level view can say "arm B converted five seeds arm A
lost and cost one it won" -- a decisive verdict from the same matches.
Chaos makes unpaired totals the weakest comparison we have and the seeds
are common random numbers by construction, so the discordant pairs ARE the
measurement ([[policy-determinism]]).

Usage::

    python -m scripts.pairs <batch_a>[:<label>] <batch_b>[:<label>]

Both batches must live under ``runs/sweeps/`` and share seeds; seeds only
one side played are reported and excluded rather than silently dropped.
The optional ``:label`` narrows a side to one arm's scorecards, which is
how an INTERLEAVED batch -- both arms on the same seeds in one directory,
the close48 shape -- compares against itself::

    python -m scripts.pairs close48:control close48:close
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from math import comb
from pathlib import Path
from typing import TypedDict

RUNS_ROOT = Path("runs/sweeps")

EXIT_OK = 0
EXIT_NO_BATCH = 1
EXIT_BAD_USAGE = 2

_VERDICT = re.compile(r"^verdict\s+(\w+)", re.MULTILINE)
_SEED = re.compile(r"-s(\d+)\.txt$")

#: Scorecard verdicts folded to the three grades a pair compares on.
GRADES: dict[str, str] = {
    "won": "W",
    "survived": "S",
    "wiped": "L",
    "defeated": "L",
}


class Pair(TypedDict):
    """One seed's verdicts under both arms.

    Attributes:
        seed: The engine seed both matches were played under.
        left: The first batch's grade, ``W``/``S``/``L``.
        right: The second batch's grade.
    """

    seed: int
    left: str
    right: str


def parse_selector(text: str) -> tuple[str, str | None]:
    """Split one side's selector into its batch and optional label.

    Args:
        text: ``"close48"`` or ``"close48:control"``.

    Returns:
        The batch name, and the label or ``None`` when the side takes every
        scorecard in the directory.
    """
    batch, colon, label = text.partition(":")
    return batch, label if colon == ":" else None


def read_grades(batch: Path, label: str | None) -> dict[int, str]:
    """Read one side's per-seed grades off its scorecards.

    Args:
        batch: The batch directory under ``runs/sweeps``.
        label: The one arm to read, or ``None`` for every scorecard. The
            narrowing is what lets two arms interleaved in one directory be
            compared: their filenames differ only in this prefix.

    Returns:
        Grade by seed, for every scorecard carrying a verdict.
    """
    pattern = "*-s*.txt" if label is None else f"{label}-s*.txt"
    grades: dict[int, str] = {}
    for card in sorted(batch.glob(pattern)):
        seed_match = _SEED.search(card.name)
        verdict_match = _VERDICT.search(card.read_text(encoding="utf-8"))
        if seed_match is None or verdict_match is None:
            continue
        grade = GRADES.get(verdict_match.group(1))
        if grade is not None:
            grades[int(seed_match.group(1))] = grade
    return grades


def pair_up(left: dict[int, str], right: dict[int, str]) -> tuple[Pair, ...]:
    """Match the two batches' grades by seed.

    Args:
        left: First batch's grades by seed.
        right: Second batch's grades by seed.

    Returns:
        One pair per shared seed, in seed order.
    """
    shared = sorted(set(left) & set(right))
    return tuple(Pair(seed=seed, left=left[seed], right=right[seed]) for seed in shared)


def format_pairs(
    name_a: str, name_b: str, pairs: Sequence[Pair], only_a: int, only_b: int
) -> tuple[str, ...]:
    """Render the paired comparison, discordant seeds first.

    Args:
        name_a: The first batch's name, for the header.
        name_b: The second batch's name.
        pairs: The shared seeds' grades.
        only_a: Seeds only the first batch played.
        only_b: Seeds only the second batch played.

    Returns:
        The lines, without newline terminators.
    """
    # Generalised from the original L->W / W->L counting: at a rung where
    # nothing loses -- Hard reads only W and S -- the L-only tally reported
    # zero flips while wins moved, which under-reported the one thing the
    # comparison exists to measure. A flip is now any move across the W
    # boundary, in either direction.
    flips_to_b = [p for p in pairs if p["left"] != "W" and p["right"] == "W"]
    flips_to_a = [p for p in pairs if p["left"] == "W" and p["right"] != "W"]
    moved = [p for p in pairs if p["left"] != p["right"]]
    discordant = len(flips_to_b) + len(flips_to_a)
    if discordant == 0:
        p_line = "p      1.000 (0 discordant pairs)"
    else:
        heavier = max(len(flips_to_b), len(flips_to_a))
        tail: int = sum(comb(discordant, i) for i in range(heavier, discordant + 1))
        # A shift, not `2**discordant`: int ** int types as Any in the stubs
        # (the sign of the exponent decides int-vs-float), and this module
        # forbids Any-typed expressions.
        ratio: float = (2 * tail) / float(1 << discordant)
        two_sided: float = ratio if ratio < 1.0 else 1.0
        p_line = f"p      {two_sided:.3f} two-sided binomial on {discordant} discordant"
    lines = [
        f"paired {len(pairs)} seed(s): {name_a} (left) vs {name_b} (right)",
        f"wins   {sum(1 for p in pairs if p['left'] == 'W')}"
        f" -> {sum(1 for p in pairs if p['right'] == 'W')}",
        f"flips  {len(flips_to_b)} to-W against {len(flips_to_a)} from-W"
        f" (net {len(flips_to_b) - len(flips_to_a):+d} for {name_b})",
        p_line,
    ]
    if only_a or only_b:
        lines.append(f"unpaired  {only_a} only in {name_a}, {only_b} only in {name_b}")
    for pair in moved:
        lines.append(f"  s{pair['seed']}: {pair['left']} -> {pair['right']}")
    return tuple(lines)


def main(argv: Sequence[str] | None = None, root: Path = RUNS_ROOT) -> int:
    """Compare two batches and print the paired verdict.

    Args:
        argv: ``<batch_a> <batch_b>``. ``None`` reads ``sys.argv[1:]``.
        root: Where batches live, a parameter so a test can point elsewhere.

    Returns:
        ``EXIT_OK`` on a printed comparison, ``EXIT_NO_BATCH`` when either
        batch is missing or empty, ``EXIT_BAD_USAGE`` on a bad argument
        count.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) != 2:
        sys.stdout.write("usage: pairs <batch_a>[:<label>] <batch_b>[:<label>]\n")
        return EXIT_BAD_USAGE
    batch_a, label_a = parse_selector(args[0])
    batch_b, label_b = parse_selector(args[1])
    grades_a = read_grades(root / batch_a, label_a)
    grades_b = read_grades(root / batch_b, label_b)
    if not grades_a or not grades_b:
        sys.stdout.write(
            f"no scorecards: {args[0]} has {len(grades_a)}, {args[1]} has {len(grades_b)}\n"
        )
        return EXIT_NO_BATCH
    pairs = pair_up(grades_a, grades_b)
    shared = {pair["seed"] for pair in pairs}
    for line in format_pairs(
        args[0],
        args[1],
        pairs,
        only_a=len(set(grades_a) - shared),
        only_b=len(set(grades_b) - shared),
    ):
        sys.stdout.write(f"{line}\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main(None))
