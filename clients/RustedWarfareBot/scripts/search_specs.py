"""The search's registered regimes: what varies between campaigns, typed.

The regime used to live in module constants inside the driver, and the
drift bit twice on one day (2026-09-02): re-aiming the base left the
champion's own value in the space (the vhsearch3 no-op arm), and aiming
the machinery at Impossible would have meant editing four constants in
step. A spec is decoded, validated at import, and chosen by name on the
command line; the driver assembles nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_dict,
    require_int,
    require_list,
    require_str,
)

from rw_bot.harness.search import SearchError
from rw_bot.policy.doctrine import INT_FIELDS


class SearchSpec(TypedDict):
    """One registered search regime -- everything that varies per campaign.

    Attributes:
        base: Repository path of the doctrine the search perturbs -- the
            rung's champion, per the campaign ledger.
        space: Alternative values per knob. Values equal to the base's
            own are filtered mechanically by ``effective_space``; every
            knob must be one of the doctrine's integer fields.
        difficulty: The AI difficulty every round plays at.
        samples: Sample budget per match.
        schedule: Pairs per candidate per round; each round keeps the
            top half of the field.
        pair_candidates: Two-knob candidates drawn per search -- law
            two's sample of the cross product the arm ladder could never
            afford.
        fitness: What the regime optimizes -- ``"margin"`` (the paired
            margin delta vs the base, every search before 2026-09-05) or
            ``"survival"`` (samples stood, the win path's Phase A figure:
            at a rung where every arm loses, the margin collapses all
            losses toward one anchor and standing time is the gradient
            [[impossible-economy-problem]]).
    """

    base: str
    space: dict[str, tuple[int, ...]]
    difficulty: int
    samples: int
    schedule: tuple[int, ...]
    pair_candidates: int
    fitness: str


def _int_values(items: Sequence[JSONValue], field: str) -> tuple[int, ...]:
    """Narrow one JSON list to integers.

    Args:
        items: The list's items, still JSON-shaped.
        field: The field name, for the error.

    Returns:
        The integers, frozen.

    Raises:
        SearchError: ``RW-SEARCH-002`` when an item is not an integer.
            Bools are refused explicitly -- Python's bool passes an int
            check, and ``True`` in a knob space would be a silent 1.
    """
    values: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SearchError(
                "RW-SEARCH-002",
                f"spec field {field!r} must hold integers, got {item!r}",
            )
        values.append(item)
    return tuple(values)


def _decode_space(raw_space: JSONObject) -> dict[str, tuple[int, ...]]:
    """Validate a regime's knob space.

    Args:
        raw_space: The space's fields, JSON-shaped.

    Returns:
        Alternative values per knob, tuples frozen, knobs sorted.

    Raises:
        SearchError: ``RW-SEARCH-002`` when a knob is not a doctrine
            integer field or has no values, or through
            :func:`_int_values` on a non-integer value.
    """
    space: dict[str, tuple[int, ...]] = {}
    for field in sorted(raw_space):
        if field not in INT_FIELDS:
            raise SearchError(
                "RW-SEARCH-002",
                f"spec space knob {field!r} is not a doctrine integer field",
            )
        values = _int_values(require_list(raw_space, field), field)
        if values == ():
            raise SearchError("RW-SEARCH-002", f"spec space knob {field!r} has no values")
        space[field] = values
    return space


def decode_search_spec(payload: JSONObject) -> SearchSpec:
    """Validate one search regime out of a plain mapping.

    Args:
        payload: The regime's fields, JSON-shaped (lists for tuples).

    Returns:
        The validated spec, tuples frozen.

    Raises:
        SearchError: ``RW-SEARCH-002`` naming the offending field when the
            base is empty, a space knob is not a doctrine integer field or
            has no values, the difficulty is outside 0-3, the samples or a
            schedule round is not positive, the schedule is empty, or the
            pair draw is negative.
        JSONTypeError: Through the ``require_*`` readers, on a missing
            field or one of the wrong shape.
    """
    base = require_str(payload, "base")
    if base == "":
        raise SearchError("RW-SEARCH-002", "spec field 'base' must name a doctrine file")
    space = _decode_space(require_dict(payload, "space"))
    difficulty = require_int(payload, "difficulty")
    if difficulty < 0 or difficulty > 3:
        raise SearchError("RW-SEARCH-002", f"spec difficulty must be 0-3, got {difficulty}")
    samples = require_int(payload, "samples")
    if samples <= 0:
        raise SearchError("RW-SEARCH-002", f"spec samples must be positive, got {samples}")
    schedule = _int_values(require_list(payload, "schedule"), "schedule")
    if schedule == ():
        raise SearchError("RW-SEARCH-002", "spec schedule must name at least one round")
    for pairs in schedule:
        if pairs <= 0:
            raise SearchError(
                "RW-SEARCH-002", f"spec schedule rounds must be positive, got {pairs}"
            )
    pair_candidates = require_int(payload, "pair_candidates")
    if pair_candidates < 0:
        raise SearchError(
            "RW-SEARCH-002",
            f"spec pair_candidates must be non-negative, got {pair_candidates}",
        )
    fitness = require_str(payload, "fitness")
    if fitness not in ("margin", "survival"):
        raise SearchError(
            "RW-SEARCH-002",
            f"spec fitness must be 'margin' or 'survival', got {fitness!r}",
        )
    return SearchSpec(
        base=base,
        space=space,
        difficulty=difficulty,
        samples=samples,
        schedule=schedule,
        pair_candidates=pair_candidates,
        fitness=fitness,
    )


#: The registered regimes, every entry validated at import. ``vh`` is the
#: Very Hard knob search around the sitting champion (evolve1-g4m2 since
#: 2026-09-04 -- the machine-learned composition, every knob inherited
#: from close0-flame4 whose neighborhood vhsearch4 already measured flat;
#: the flame axis history: the 2 -> 4 adoption measured a strong gradient
#: with the old cap sitting on the champion).
#: ``imp`` aims the same machinery at Impossible's untried
#: composed fortress vocabulary: every value has been fielded in a
#: committed doctrine (guns 1-2 from the zone arms, nukes 1 from
#: fortress-nuke, mass 40 from the fortress chassis, strike 5000/15000
#: from the release arms, close 6 from the adopted latch) -- the singles
#: were each rejected at 3-6 seed screens in the pre-cluster era with the
#: economy named as the blocker, and the cluster makes re-asking the
#: COMPOSED question affordable, margin-triaged, where wins are zero.
#: ``income_ladder``, ``cover`` and ``riposte`` are deliberately absent:
#: they are flag fields, and the candidate machinery moves integers.
SPECS: Mapping[str, SearchSpec] = {
    "vh": decode_search_spec(
        {
            "base": "doctrines/evolve1-g4m2.doctrine",
            "space": {
                "flame": [0, 2, 4, 6, 8],
                "close": [0, 6],
                "raid": [0, 6],
                "tech": [0, 2],
                "medics": [1],
                "decoys": [2],
            },
            "difficulty": 2,
            "samples": 10000,
            "schedule": [8, 16],
            "pair_candidates": 6,
            "fitness": "margin",
        }
    ),
    "imp": decode_search_spec(
        {
            "base": "doctrines/flame-nocover.doctrine",
            "space": {
                "guns": [1, 2],
                "nukes": [1],
                "close": [6],
                "mass": [40],
                "strike": [5000, 15000],
                "tech": [0, 2],
            },
            "difficulty": 3,
            "samples": 10000,
            "schedule": [8, 16],
            "pair_candidates": 6,
            "fitness": "margin",
        }
    ),
    # The win path's Phase A regime ([[impossible-economy-problem]]):
    # survival, not margin -- at a rung with zero wins the margin collapses
    # every loss toward one anchor, and standing time is the gradient the
    # turtle-bank-nuke chain needs first. Re-aimed 2026-09-06 at evolve3's
    # own graduate (+510.6 paired survival at t=4.20 on its first panel):
    # the iterated climb searches AROUND the current best stander, the
    # same re-aiming the vh regime performed at each adoption.
    "imp-survival": decode_search_spec(
        {
            "base": "doctrines/evolve3-g3m10.doctrine",
            "space": {
                "guns": [1, 2],
                "nukes": [1],
                "close": [6],
                "mass": [40],
                "strike": [5000, 15000],
                "tech": [0, 2],
            },
            "difficulty": 3,
            "samples": 10000,
            "schedule": [8, 16],
            "pair_candidates": 6,
            "fitness": "survival",
        }
    ),
}


def require_search_spec(name: str) -> SearchSpec:
    """Resolve a registered regime by name.

    Args:
        name: The regime's key in :data:`SPECS`.

    Returns:
        The spec.

    Raises:
        SearchError: ``RW-SEARCH-002`` naming the known regimes when the
            name is not one of them.
    """
    spec = SPECS.get(name)
    if spec is None:
        known = ", ".join(sorted(SPECS))
        raise SearchError("RW-SEARCH-002", f"unknown search spec {name!r}; registered: {known}")
    return spec


__all__ = [
    "SPECS",
    "SearchSpec",
    "decode_search_spec",
    "require_search_spec",
]
