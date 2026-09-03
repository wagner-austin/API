"""Compiling a continuous genome into an ordinary doctrine.

The population search's whole trick (``[[harness-population-search]]``)
is that a candidate is a point in a continuous space -- army-composition
weights on a simplex plus integer knobs -- but the thing evaluated is a
plain doctrine file, so the entire measurement chain (payload freeze,
campaign arrays, scorecards, paired margin) runs unchanged. This module
is that boundary: weights in, doctrine out, deterministic.

Two mechanics facts shape the design and both were read out of the
policy code rather than assumed. First, ``goals`` repeats are a RATIO
(production holds the army to the mix forever), so weights become repeat
counts. Second, ``goals`` is also the planner's OPENING BUILD ORDER, so
the compiled tail keeps exactly as many army slots as the base doctrine
carries -- a genome changes what the slots hold, never how many there
are, or every candidate would also be an opening-length experiment
nobody asked for.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final

from rw_bot import RwBotError
from rw_bot.policy.doctrine import INT_FIELDS, Doctrine
from rw_bot.policy.doctrine_codecs import decode_doctrine, encode_doctrine

#: The ground units a v1 genome may weight, every one already fielded in
#: a committed doctrine's goals. Sorted, because ties in the apportionment
#: break by this order and a set would make compilation nondeterministic.
ARMY_VOCABULARY: Final = ("c_artillery", "c_tank", "heavyTank", "hoverTank")


class GenomeError(RwBotError):
    """A genome could not be compiled into a doctrine.

    Args:
        code: ``RW-GENOME-001`` for a weight outside the vocabulary or an
            invalid weight value, ``RW-GENOME-002`` for a base doctrine
            with no army slots to apportion, ``RW-GENOME-003`` for a knob
            outside the doctrine's integer fields.
        message: Human-readable description of the refusal.
    """


def army_split(goals: tuple[str, ...]) -> tuple[tuple[str, ...], int]:
    """Divide a goals list into its scaffold prefix and its army slots.

    Args:
        goals: The base doctrine's goals, in build order.

    Returns:
        The entries outside :data:`ARMY_VOCABULARY` in their original
        order (the economy scaffold -- extractors and kin), and the
        count of entries inside it (the army slots a genome fills).
    """
    scaffold = tuple(entry for entry in goals if entry not in ARMY_VOCABULARY)
    return scaffold, len(goals) - len(scaffold)


def ratio_slots(weights: Mapping[str, float], slots: int) -> tuple[str, ...]:
    """Apportion army slots to units by largest remainder.

    Deterministic by construction: quotas are ``weight / total * slots``,
    every unit takes its quota's floor, and the leftover slots go to the
    largest fractional remainders with ties broken by vocabulary order.
    The result lists each unit's repeats consecutively, higher counts
    first, count ties again by vocabulary order.

    Args:
        weights: Non-negative weight per unit; keys must be vocabulary
            members and at least one weight must be positive.
        slots: How many entries to fill, at least one.

    Returns:
        ``slots`` unit names.

    Raises:
        GenomeError: ``RW-GENOME-001`` when a key is outside the
            vocabulary, a weight is negative or non-finite, or no weight
            is positive -- a genome that cannot say what army it wants
            has no doctrine to compile.
    """
    total = 0.0
    for unit in sorted(weights):
        if unit not in ARMY_VOCABULARY:
            raise GenomeError(
                "RW-GENOME-001", f"weight names a unit outside the vocabulary: {unit!r}"
            )
        value = weights[unit]
        if not math.isfinite(value) or value < 0.0:
            raise GenomeError(
                "RW-GENOME-001",
                f"weight for {unit!r} must be finite and non-negative, got {value!r}",
            )
        total += value
    if total <= 0.0:
        raise GenomeError("RW-GENOME-001", "no weight is positive; the genome names no army")

    quotas = {unit: weights.get(unit, 0.0) / total * slots for unit in ARMY_VOCABULARY}
    counts = {unit: int(quotas[unit]) for unit in ARMY_VOCABULARY}
    leftover = slots - sum(counts.values())

    def remainder_rank(unit: str) -> tuple[float, str]:
        return (counts[unit] - quotas[unit], unit)

    for unit in sorted(ARMY_VOCABULARY, key=remainder_rank)[:leftover]:
        counts[unit] += 1

    def count_rank(unit: str) -> tuple[int, str]:
        return (-counts[unit], unit)

    ordered = sorted(ARMY_VOCABULARY, key=count_rank)
    tail: list[str] = []
    for unit in ordered:
        tail.extend([unit] * counts[unit])
    return tuple(tail)


def compile_genome(
    base: Doctrine, weights: Mapping[str, float], knobs: Mapping[str, int], name: str
) -> Doctrine:
    """Build the doctrine one genome describes.

    Args:
        base: The champion the population perturbs; its economy scaffold,
            flag fields, and every knob the genome does not move are
            carried unchanged.
        weights: Army-composition weights, per :func:`ratio_slots`.
        knobs: Integer knob overrides, keys inside ``INT_FIELDS``.
        name: The candidate's label, recorded as the doctrine's name.

    Returns:
        The compiled doctrine, validated by the ordinary codec.

    Raises:
        GenomeError: ``RW-GENOME-001`` through :func:`ratio_slots`,
            ``RW-GENOME-002`` when the base carries no army slots,
            ``RW-GENOME-003`` when a knob is outside the doctrine's
            integer fields.
        DecodeError: When a knob value is outside its codec range,
            through the doctrine codec's ordinary validation.
    """
    scaffold, slots = army_split(base["goals"])
    if slots == 0:
        raise GenomeError(
            "RW-GENOME-002",
            "the base doctrine's goals carry no army slots to apportion",
        )
    payload = dict(encode_doctrine(base))
    for field in sorted(knobs):
        if field not in INT_FIELDS:
            raise GenomeError("RW-GENOME-003", f"unknown doctrine knob: {field}")
        payload[field] = knobs[field]
    payload["goals"] = ",".join((*scaffold, *ratio_slots(weights, slots)))
    payload["name"] = name
    return decode_doctrine(payload)


__all__ = [
    "ARMY_VOCABULARY",
    "GenomeError",
    "army_split",
    "compile_genome",
    "ratio_slots",
]
