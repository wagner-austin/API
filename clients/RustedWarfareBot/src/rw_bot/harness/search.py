"""Successive halving over doctrine variants, scored by the dense margin.

The arm ladder was manual coordinate descent: one knob, one evening, one
panel. Eleven-plus rejected arms at Very Hard say the single-knob
vocabulary is mined out, and law two says combinations are their own
measurements -- which makes the search space too large for hand panels.
This module is the pure half of the automated walk: candidate variants
from a knob space, margin-scored rounds, and a halving rule that spends
matches on survivors.

The discipline the laws impose stays explicit: the margin
([[rw_bot.harness.margin]]) is triage only. Whatever survives the last
round graduates to an ordinary full panel judged on wins against the +4
bar, and then fresh-tree replication (laws six and nine). The search
proposes; the bar disposes.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from rw_bot import RwBotError
from rw_bot.policy.doctrine import INT_FIELDS, Doctrine
from rw_bot.policy.doctrine_codecs import decode_doctrine, encode_doctrine

#: One candidate: the knob moves it applies to the base doctrine, sorted
#: by field so equal candidates compare equal.
Candidate = tuple[tuple[str, int], ...]


class SearchError(RwBotError):
    """A candidate asked for a knob the doctrine does not carry.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description naming the knob.
    """


def candidate_label(moves: Candidate) -> str:
    """Name a candidate by its moves, stable across runs.

    Args:
        moves: The knob moves.

    Returns:
        ``"raid5-tech2"`` for ``(("raid", 5), ("tech", 2))``.
    """
    return "-".join(f"{field}{value}" for field, value in moves)


def single_moves(space: Mapping[str, Sequence[int]]) -> tuple[Candidate, ...]:
    """Every one-knob candidate the space allows, in field order.

    Args:
        space: Alternative values per knob; the caller keeps the base
            doctrine's own values out of the space.

    Returns:
        One candidate per (field, value).
    """
    return tuple(((field, value),) for field in sorted(space) for value in space[field])


def effective_space(
    space: Mapping[str, Sequence[int]], base: Doctrine
) -> dict[str, tuple[int, ...]]:
    """Drop each knob's base value from its alternatives.

    :func:`single_moves`' contract -- the caller keeps the base doctrine's
    own values out of the space -- was maintained by hand until 2026-09-02,
    when re-aiming the search at flame-close6 left ``close 6`` in the space
    and vhsearch3 fielded a no-op arm: doctrine-identical to control, it
    measured pure label-noise, survived round 0 on it, and burned sixteen
    round-1 pairs. The filter is mechanical now; a re-aimed base can no
    longer silently turn a candidate into a coin toss.

    Args:
        space: Alternative values per knob.
        base: The champion the search perturbs.

    Returns:
        The space with every value equal to the base's own dropped, and
        any knob with nothing left removed entirely.

    Raises:
        SearchError: ``RW-SEARCH-001`` when a knob is outside the
            doctrine's integer fields -- a knob that cannot be perturbed
            cannot be filtered either.
    """
    payload = dict(encode_doctrine(base))
    filtered: dict[str, tuple[int, ...]] = {}
    for field in sorted(space):
        if field not in INT_FIELDS:
            raise SearchError("RW-SEARCH-001", f"unknown doctrine knob: {field}")
        values = tuple(value for value in space[field] if value != payload[field])
        if values != ():
            filtered[field] = values
    return filtered


def sampled_pairs(
    space: Mapping[str, Sequence[int]], count: int, seed: int
) -> tuple[Candidate, ...]:
    """A deterministic sample of two-knob candidates.

    Law two: knobs do not compose freely, so pairs are their own
    measurements -- but the full cross product is unaffordable, so a
    seeded sample stands in for it, reproducible from the seed alone.

    Args:
        space: Alternative values per knob.
        count: How many pairs to draw, capped by how many exist.
        seed: The sample's reproducibility anchor.

    Returns:
        Distinct two-knob candidates, each combining different fields.
    """
    fields = sorted(space)
    pool: list[Candidate] = []
    for i, first in enumerate(fields):
        for second in fields[i + 1 :]:
            for a in space[first]:
                for b in space[second]:
                    pool.append(((first, a), (second, b)))
    if count >= len(pool):
        return tuple(pool)
    return tuple(random.Random(seed).sample(pool, count))


def apply_moves(base: Doctrine, moves: Candidate) -> Doctrine:
    """Build the variant doctrine one candidate names.

    Args:
        base: The champion doctrine the search perturbs.
        moves: The knob moves to apply.

    Returns:
        A copy of the base with the moves applied and the candidate's
        label as its name.

    Raises:
        SearchError: ``RW-SEARCH-001`` when a move names a knob outside
            the doctrine's integer fields -- a typo in a knob space must
            stop the search, not silently perturb nothing.
        DecodeError: When a move's value is outside the knob's own range,
            through the doctrine codec's ordinary validation.
    """
    payload = dict(encode_doctrine(base))
    for field, value in moves:
        if field not in INT_FIELDS:
            raise SearchError("RW-SEARCH-001", f"unknown doctrine knob: {field}")
        payload[field] = value
    payload["name"] = candidate_label(moves)
    return decode_doctrine(payload)


def paired_delta(
    margins: Mapping[str, Mapping[int, float]], arm: str, control: str
) -> tuple[int, float, float]:
    """Summarize one arm's paired margin advantage over a control.

    Args:
        margins: Margins by arm and seed, as ``batch_margins`` returns.
        arm: The candidate arm's label.
        control: The control arm's label.

    Returns:
        ``(pairs, mean delta, sd)``; zeroes when no seeds are shared.
    """
    ours = margins.get(arm, {})
    base = margins.get(control, {})
    shared = sorted(set(ours) & set(base))
    if not shared:
        return 0, 0.0, 0.0
    deltas = [ours[s] - base[s] for s in shared]
    mean = sum(deltas) / len(deltas)
    sd = math.sqrt(sum((d - mean) * (d - mean) for d in deltas) / len(deltas))
    return len(shared), mean, sd


def keep_top(scores: Mapping[Candidate, float], keep: int) -> tuple[Candidate, ...]:
    """The survivors of one halving round, deterministically ordered.

    Args:
        scores: Mean paired margin delta per candidate.
        keep: How many survive.

    Returns:
        The best ``keep`` candidates, highest delta first, ties broken
        by label so two runs of one search agree.
    """

    def rank(candidate: Candidate) -> tuple[float, str]:
        return (-scores[candidate], candidate_label(candidate))

    return tuple(sorted(scores, key=rank)[: max(0, keep)])


__all__ = [
    "Candidate",
    "SearchError",
    "apply_moves",
    "candidate_label",
    "effective_space",
    "keep_top",
    "paired_delta",
    "sampled_pairs",
    "single_moves",
]
