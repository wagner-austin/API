"""One match's result, read off the scorecard it was filed as.

A batch's store is its scorecards: one text file per match, written by
:func:`~rw_bot.harness.runner.play_job` and never rewritten. Everything
downstream -- the per-batch table, the per-arm aggregate, the run record --
reads them back, so what a row IS belongs here rather than in whichever
reader happened to need it first.

IT WAS IN WHICHEVER READER NEEDED IT FIRST. ``scripts/analyze_sweep`` carried
its own copy of the label/value reader that
:func:`~rw_bot.harness.margin.scorecard_fields` already was, differing only in
``line[0]`` versus ``line[:1]``. Two readers of one format is one edit away
from a table and an aggregate that disagree about the same batch, and neither
would fail -- they would simply report different numbers for the same match.

THE ROW IS TYPED because it is a payload, not a scratch dictionary. It used to
be ``dict[str, str | int]``, which meant every consumer narrowed each field
itself and a misspelled key read as a missing figure rather than an error.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypedDict

from rw_bot.harness.margin import scorecard_fields
from rw_bot.validation import require_int, require_non_empty_str, require_str

#: Separator between a match's arm and its seed in a result's filename.
#:
#: Split from the RIGHT: an arm label may itself contain the separator --
#: ``aa-counter-s2`` is a real arm name -- and splitting from the left would
#: file it under ``aa-counter`` with a seed of ``2``.
SEED_MARKER = "-s"

#: Reads the end figure out of a ``start -> end`` scorecard value.
_ARROW_END = re.compile(r"->\s*(-?\d+)")

#: Reads the worst dip out of the ``best rival`` line.
_WORST_DIP = re.compile(r"worst dip (\d+)")

#: Reads how many of the enemies seen could actually be fought.
_ENGAGEABLE = re.compile(r"\((\d+) engageable\)")


class MatchRow(TypedDict):
    """What one match is, as the figures every reader of a batch wants.

    Attributes:
        arm: Which arm played it, from the result's filename.
        seed: What the engine's generator was pinned to.
        verdict: How the match ended, first word only.
        extr_end: Extractors standing at the end.
        peak: The most extractors held at once, from the trace.
        dropped: How many of that peak were gone by the end. The figure every
            verdict this project has produced turns on.
        worth_end: Total worth at the end.
        rival_end: The strongest rival's worth at the end.
        dip: The worst the rival was ever driven down by.
        targets_end: Enemies visible at the end.
        engageable: How many of those could be fought at all.
        intercepted: Interceptions made.
        income: Income at the end, as the scorecard renders it.
    """

    arm: str
    seed: int
    verdict: str
    extr_end: int
    peak: int
    dropped: int
    worth_end: int
    rival_end: int
    dip: int
    targets_end: int
    engageable: int
    intercepted: int
    income: str


def split_stem(stem: str) -> tuple[str, int]:
    """Read the arm and seed a result's filename names.

    Args:
        stem: A result filename without its suffix, e.g. ``attack-s12345``.

    Returns:
        The arm and the seed.

    Raises:
        ValueError: When the stem carries no seed marker, or the seed is not a
            whole number. Refused rather than guessed: a result whose arm
            cannot be read would be aggregated into the wrong one, and an arm
            is what a verdict is about.
    """
    arm, marker, seed = stem.rpartition(SEED_MARKER)
    if marker == "":
        raise ValueError(f"a result filename carries {SEED_MARKER!r} before its seed, got {stem!r}")
    if not (seed.lstrip("-").isdigit() and seed.lstrip("-") != ""):
        raise ValueError(f"a result filename ends with a whole seed, got {stem!r}")
    return arm, int(seed)


def arrow_end(value: str) -> int:
    """Return the end figure of a ``start -> end`` scorecard value.

    Args:
        value: The raw field text.

    Returns:
        The end integer, zero when the shape is absent -- a figure the match
        never recorded is zero rather than an error, because a scorecard from
        an older build legitimately lacks lines a newer one writes.
    """
    found = _ARROW_END.search(value)
    return int(found.group(1)) if found else 0


def parse_match_row(stem: str, text: str, peak: int, dropped: int) -> MatchRow:
    """Read one match's row from its scorecard.

    Args:
        stem: The result's filename without its suffix.
        text: The scorecard's content.
        peak: Most extractors held at once, read from the match's trace.
        dropped: How many of that peak were gone by the end, likewise.

    Returns:
        The row.

    Raises:
        ValueError: When the filename does not name an arm and a seed.
    """
    arm, seed = split_stem(stem)
    fields = scorecard_fields(text)
    rival = fields.get("best rival", "")
    dip = _WORST_DIP.search(rival)
    enemies = fields.get("enemies seen", "")
    engageable = _ENGAGEABLE.search(enemies)
    return MatchRow(
        arm=arm,
        seed=seed,
        verdict=fields.get("verdict", "?").split(" ")[0],
        extr_end=arrow_end(fields.get("extractors", "")),
        peak=peak,
        dropped=dropped,
        worth_end=arrow_end(fields.get("total worth", "")),
        rival_end=arrow_end(rival.split("(")[0]) if rival else 0,
        dip=int(dip.group(1)) if dip else 0,
        targets_end=arrow_end(enemies.split("(")[0]) if enemies else 0,
        engageable=int(engageable.group(1)) if engageable else 0,
        intercepted=int(fields.get("intercepted", "0") or 0),
        income=fields.get("income", "?"),
    )


def row_order(row: MatchRow) -> tuple[str, int]:
    """Return the key a batch's rows are ordered by.

    Arm then seed, so a table and a record built from the same batch list its
    matches identically -- and so neither changes with the order the
    filesystem happened to list the results in.

    Args:
        row: One match.

    Returns:
        The sort key.
    """
    return (row["arm"], row["seed"])


def decode_match_row(payload: Mapping[str, str | int | float | bool]) -> MatchRow:
    """Read a match row from a flat payload.

    Args:
        payload: Field values by name.

    Returns:
        The row.

    Raises:
        DecodeError: ``RW-DECODE-001`` when a field is absent, ``RW-DECODE-002``
            when one carries the wrong type, ``RW-DECODE-003`` when the arm or
            verdict is blank.
    """
    return MatchRow(
        arm=require_non_empty_str(payload, "arm"),
        seed=require_int(payload, "seed"),
        verdict=require_non_empty_str(payload, "verdict"),
        extr_end=require_int(payload, "extr_end"),
        peak=require_int(payload, "peak"),
        dropped=require_int(payload, "dropped"),
        worth_end=require_int(payload, "worth_end"),
        rival_end=require_int(payload, "rival_end"),
        dip=require_int(payload, "dip"),
        targets_end=require_int(payload, "targets_end"),
        engageable=require_int(payload, "engageable"),
        intercepted=require_int(payload, "intercepted"),
        income=require_str(payload, "income"),
    )


def encode_match_row(row: MatchRow) -> dict[str, str | int]:
    """Write a match row back to a flat payload.

    Args:
        row: The row.

    Returns:
        Field values by name, as :func:`decode_match_row` reads them.
    """
    return {
        "arm": row["arm"],
        "seed": row["seed"],
        "verdict": row["verdict"],
        "extr_end": row["extr_end"],
        "peak": row["peak"],
        "dropped": row["dropped"],
        "worth_end": row["worth_end"],
        "rival_end": row["rival_end"],
        "dip": row["dip"],
        "targets_end": row["targets_end"],
        "engageable": row["engageable"],
        "intercepted": row["intercepted"],
        "income": row["income"],
    }


__all__ = [
    "SEED_MARKER",
    "MatchRow",
    "arrow_end",
    "decode_match_row",
    "encode_match_row",
    "parse_match_row",
    "row_order",
    "split_stem",
]
