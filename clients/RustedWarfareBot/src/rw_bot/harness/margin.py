"""The dense margin: how much a match was won or lost by, not just whether.

A win/loss is one bit and 48 pairs of one bit is a blunt instrument; the
scorecards already carry the erosion law's figures ([[policy-trace]], law
seven: endpoints lie, peaks tell), and this reader turns them into one
bounded number per match:

    margin = verdict + pressure + tempo

* **verdict** anchors the ordering: won +2, survived +1, defeated -1,
  wiped -2.
* **pressure** is the fraction of the best rival's peak worth that was
  destroyed -- ``worst dip / peak`` from the card's ``best rival`` line,
  in [0, 1]. A defeat that gutted the enemy economy is closer to a win
  than one that never drew blood.
* **tempo** rewards fast wins and long losses: ``1 - t`` when won,
  ``t - 1`` when defeated or wiped, 0 when survived, where ``t`` is the
  match's samples over the batch's longest match. Games are decided by
  tempo (the naval verdict's own lesson, log 2026-08-11), so a margin
  that ignored it would miss what the panels keep measuring.

The bands cannot cross: any wiped (max -1) < any defeated (max 0) < any
survived (min 1) < any won (min 2), so ranking by margin never disagrees
with ranking by verdict -- margin only separates matches the verdict
calls equal.

The margin is an analysis layer only: verdicts and the win bar still
rule adoption. ``scripts/margin.py`` is the CLI over these functions;
the doctrine-search driver reads them directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from rw_bot.harness.sweep import LABEL_WIDTH

#: The verdict's anchor scores, ordering preserved by construction.
VERDICT_SCORES: Mapping[str, float] = {
    "won": 2.0,
    "survived": 1.0,
    "defeated": -1.0,
    "wiped": -2.0,
}


def scorecard_fields(text: str) -> dict[str, str]:
    """Read a scorecard's label/value pairs by the shape the sweep trusts.

    Args:
        text: The scorecard file's content.

    Returns:
        Values by label.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) > LABEL_WIDTH and line[LABEL_WIDTH] != " " and line[:1].islower():
            out[line[:LABEL_WIDTH].strip()] = line[LABEL_WIDTH:].strip()
    return out


def pressure_of(best_rival: str) -> float:
    """Read the destroyed fraction of the enemy's peak from the card line.

    Args:
        best_rival: The ``best rival`` value, shaped
            ``"3500 -> 139900 (peak 139900, worst dip 11600)"``.

    Returns:
        ``worst dip / peak`` clamped to [0, 1], or 0.0 when the line does
        not carry both figures -- an unreadable line reads as no pressure
        rather than crashing a whole batch's report.
    """
    if "peak " not in best_rival or "worst dip " not in best_rival:
        return 0.0
    peak_text = best_rival.split("peak ")[1].split(",")[0]
    dip_text = best_rival.split("worst dip ")[1].split(")")[0]
    if not peak_text.isdigit() or not dip_text.isdigit():
        return 0.0
    peak = int(peak_text)
    if peak <= 0:
        return 0.0
    return min(1.0, max(0.0, int(dip_text) / peak))


def margin_of(verdict: str, pressure: float, samples: int, longest: int) -> float | None:
    """Score one match on the bounded margin scale.

    Args:
        verdict: The card's verdict word.
        pressure: Destroyed fraction of the enemy peak, [0, 1].
        samples: This match's samples seen.
        longest: The batch's longest match, the tempo normalizer.

    Returns:
        The margin, or None for a verdict outside the four bands --
        an unfinished card is not a measurement.
    """
    anchor = VERDICT_SCORES.get(verdict)
    if anchor is None:
        return None
    t = min(1.0, samples / longest) if longest > 0 else 1.0
    if verdict == "won":
        tempo = 1.0 - t
    elif verdict in ("defeated", "wiped"):
        tempo = t - 1.0
    else:
        tempo = 0.0
    return anchor + pressure + tempo


def batch_margins(batch_dir: Path) -> dict[str, dict[int, float]]:
    """Score every completed match in one batch directory.

    Args:
        batch_dir: The sweep directory holding ``<arm>-s<seed>.txt`` cards.

    Returns:
        Margins by arm, then by seed.

    Raises:
        OSError: When a card cannot be read.
    """
    cards: list[tuple[str, int, dict[str, str]]] = []
    longest = 0
    for card_path in sorted(batch_dir.glob("*-s*.txt")):
        arm, _, seed_text = card_path.stem.rpartition("-s")
        if not arm or not seed_text.isdigit():
            continue
        fields = scorecard_fields(card_path.read_text(encoding="utf-8"))
        samples_text = fields.get("samples seen", "")
        samples = int(samples_text) if samples_text.isdigit() else 0
        longest = max(longest, samples)
        cards.append((arm, int(seed_text), fields))
    margins: dict[str, dict[int, float]] = {}
    for arm, seed, fields in cards:
        verdict = fields.get("verdict", "").split(" ")[0]
        samples_text = fields.get("samples seen", "")
        samples = int(samples_text) if samples_text.isdigit() else 0
        margin = margin_of(verdict, pressure_of(fields.get("best rival", "")), samples, longest)
        if margin is not None:
            margins.setdefault(arm, {})[seed] = margin
    return margins


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def report(batch: str, margins: Mapping[str, Mapping[int, float]]) -> tuple[str, ...]:
    """Render one batch's margin summary.

    Args:
        batch: The batch name, for the heading.
        margins: Margins by arm and seed.

    Returns:
        Report lines: one per arm, then one per arm pair sharing seeds,
        with the paired margin delta beside the paired win delta.
    """
    lines = [f"## {batch}"]
    for arm in sorted(margins):
        scores = list(margins[arm].values())
        wins = sum(1 for s in scores if s >= 2.0)
        lines.append(
            f"{arm:12} n={len(scores):3}  mean margin {_mean(scores):+.3f}"
            f"  wins {wins}/{len(scores)}"
        )
    arms = sorted(margins)
    for i, base in enumerate(arms):
        for other in arms[i + 1 :]:
            shared = sorted(set(margins[base]) & set(margins[other]))
            if not shared:
                continue
            deltas = [margins[other][s] - margins[base][s] for s in shared]
            win_delta = sum(
                (1 if margins[other][s] >= 2.0 else 0) - (1 if margins[base][s] >= 2.0 else 0)
                for s in shared
            )
            centre = _mean(deltas)
            sd = math.sqrt(_mean([(d - centre) * (d - centre) for d in deltas]))
            lines.append(
                f"paired {other} - {base}: n={len(shared)}"
                f"  margin delta {_mean(deltas):+.3f} (sd {sd:.3f})"
                f"  win delta {win_delta:+d}"
            )
    return tuple(lines)


__all__ = [
    "VERDICT_SCORES",
    "batch_margins",
    "margin_of",
    "pressure_of",
    "report",
    "scorecard_fields",
]
