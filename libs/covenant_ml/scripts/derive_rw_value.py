"""Derive the ``rw_value`` regression corpus from ``rw_matches``.

``rw_matches`` is the outcome corpus: one row per recorded sample of a
RustedWarfare match, labelled ``won``. This script derives the VALUE
corpus from the same rows: the target is ``frames_remaining`` — how many
frames separate this sample from its match's last recorded sample — a
time-to-verdict regression that asks the model to read a position's
distance from resolution out of the same mid-match state the outcome
model reads.

Honesty rules, applied by construction:

- ``won`` and ``verdict`` are DROPPED: a value model that knows the final
  outcome is answering an easier question than the one deployment asks.
- ``arm``, ``seed`` and ``difficulty`` are dropped exactly as
  ``rw_matches`` excludes them: run identity and setup, not state.
- ``match`` is kept as the group column, never a feature: 1,500 rows of
  one match agree with each other far more than with anything else, so
  the split must be by match — the same law the outcome corpus follows.
- The end frame is each match's own last recorded frame. Nothing is
  imputed: a match's final sample has ``frames_remaining = 0`` by
  definition, not by assumption.

Usage:
    poetry run python -m scripts.derive_rw_value \
        --source ../../services/covenant-radar-api/data/external/rw_matches/data.csv \
        --out ../../services/covenant-radar-api/data/external/rw_value/data.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TypedDict

#: Columns copied through as features, in output order. ``match`` leads as
#: the group column; ``frame`` stays a feature (elapsed time is state the
#: bot observes, and knowing the clock does not reveal the end).
FEATURE_COLUMNS = (
    "frame",
    "army",
    "credits",
    "enemies",
    "extractors",
    "lost",
    "lost_cum",
    "killed_cum",
    "producers",
    "idle",
    "orders",
    "refused",
    "worth",
    "rival",
    "income",
    "rival_income",
)

#: Columns the source carries that the value corpus must not: outcome
#: restatements and run identity/setup.
DROPPED_COLUMNS = ("arm", "seed", "verdict", "won", "difficulty")

TARGET_COLUMN = "frames_remaining"


class DerivedCorpus(TypedDict):
    """The derived value corpus, ready to write.

    Args:
        header: Output column names in order: ``match``, the features,
            then the target.
        rows: One output row per source row, values as strings in header
            order.
        n_matches: Distinct match count.
        target_mean: Mean of the target column.
        target_max: Maximum of the target column.
    """

    header: tuple[str, ...]
    rows: list[list[str]]
    n_matches: int
    target_mean: float
    target_max: int


def _write(message: str) -> None:
    """Write a message to stdout.

    Args:
        message: Text to emit.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Derive the rw_value time-to-verdict corpus from rw_matches."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the rw_matches data.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the rw_value data.csv.",
    )
    return parser


def derive_rw_value(source: Path) -> DerivedCorpus:
    """Derive the value corpus from the outcome corpus file.

    Two passes over the source rows: the first finds each match's last
    recorded frame, the second emits every row with its distance from
    that end.

    Args:
        source: Path to the rw_matches CSV.

    Returns:
        The derived corpus and its summary statistics.

    Raises:
        ValueError: If the source is missing a required column.
    """
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        source_header = next(reader)
        source_rows = [row for row in reader if row]

    column_index: dict[str, int] = {name: i for i, name in enumerate(source_header)}
    for name in ("match", *FEATURE_COLUMNS):
        if name not in column_index:
            raise ValueError(f"rw_matches source is missing required column '{name}'")

    match_idx = column_index["match"]
    frame_idx = column_index["frame"]

    end_frames: dict[str, int] = {}
    for row in source_rows:
        match_id = row[match_idx]
        frame = int(row[frame_idx])
        recorded = end_frames.get(match_id)
        if recorded is None or frame > recorded:
            end_frames[match_id] = frame

    header = ("match", *FEATURE_COLUMNS, TARGET_COLUMN)
    rows: list[list[str]] = []
    target_sum = 0
    target_max = 0
    for row in source_rows:
        match_id = row[match_idx]
        remaining = end_frames[match_id] - int(row[frame_idx])
        target_sum += remaining
        target_max = max(target_max, remaining)
        out_row = [match_id]
        for name in FEATURE_COLUMNS:
            out_row.append(row[column_index[name]])
        out_row.append(str(remaining))
        rows.append(out_row)

    n_rows = len(rows)
    return DerivedCorpus(
        header=header,
        rows=rows,
        n_matches=len(end_frames),
        target_mean=target_sum / n_rows if n_rows > 0 else 0.0,
        target_max=target_max,
    )


def main(argv: list[str] | None = None) -> int:
    """Derive the corpus, write it, and report its shape.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parsed = build_parser().parse_args(argv)
    source: Path = parsed.source
    out: Path = parsed.out

    derived = derive_rw_value(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(derived["header"])
        writer.writerows(derived["rows"])

    _write(
        f"rw_value: {len(derived['rows'])} rows across {derived['n_matches']} matches -> {out}\n"
        f"  {TARGET_COLUMN}: mean {derived['target_mean']:.1f}, "
        f"max {derived['target_max']} frames\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
