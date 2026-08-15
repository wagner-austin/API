"""Standing threat as a queryable surface: who is under whose guns, now.

The spatial layer's first slice ([[policy-threat]] names the gap: threat
answers one route at a time, and nothing the bot records says how much of
what it OWNS stands inside hostile reach). The campaign's losses read
"overrun with the economy standing" and the doom model's ninety features
are all counts and rates -- not one of them knows where anything is
(log 2026-08-09, the enemy-shape lesson, one dimension up).

Everything here is the engine's own arithmetic composed, never a number
this module invented: hostility is the engine's alliance predicate on the
wire, whether a gun reaches a layer is the engine's own test transcribed
in :func:`~rw_bot.mechanics.combat_profile.can_engage`, and reach is the
registry's declared attack range ([[mechanics-combat-profile]]). Coverage
is judged where units stand THIS sample -- no closure model, no path
model, the same deliberate scope [[policy-threat]] records.

The first consumer is the trace ([[policy-trace]]): three columns per
sample, feeding the next model refit -- a recording, not a behavior, so
law five (input without a gated response is perturbation) is not
provoked. A positioning consumer reads the same functions when one earns
its arm.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from rw_bot.mechanics.combat_profile import CombatProfile, can_engage, reach_of
from rw_bot.wire.state import Entity, Sample


class Coverage(TypedDict):
    """How much of each side stands inside the other's standing reach.

    Attributes:
        eco_covered: Finished extractors of ours with at least one visible
            hostile gun covering them -- the "decided by extractors LOST"
            figure ([[policy-economy]]) as a per-sample spatial read.
        own_covered: Owned complete entities with at least one visible
            hostile gun covering them, structures and army alike.
        foe_covered: Visible hostiles standing inside at least one of our
            own guns' reach -- the reverse direction, so the PAIR carries
            the engagement balance a single count cannot.
    """

    eco_covered: int
    own_covered: int
    foe_covered: int


def guns_covering(
    sample: Sample,
    profiles: Mapping[str, CombatProfile],
    target: Entity,
) -> int:
    """Count the visible hostiles whose guns cover the target where it stands.

    A gun covers the target when its weapon reaches the target's layer --
    the engine's own test -- AND its declared reach spans the distance
    right now. Unarmed hostiles fall out through zero reach.

    Args:
        sample: One observation of the world.
        profiles: Combat profiles by type name, from the registry dump.
        target: The entity being covered.

    Returns:
        How many hostile guns cover it, zero for safe ground.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not
            describe a visible type -- a stale dump, not safe ground.
    """
    count = 0
    for entity in sample["entities"]:
        if not entity["hostile"]:
            continue
        if not can_engage(profiles, entity, target):
            continue
        reach = reach_of(profiles, entity)
        if reach <= 0.0:
            continue
        if (entity["x"] - target["x"]) ** 2 + (entity["y"] - target["y"]) ** 2 <= reach * reach:
            count += 1
    return count


def coverage(
    sample: Sample,
    profiles: Mapping[str, CombatProfile],
    extractor_family: frozenset[str],
) -> Coverage:
    """Measure both directions of standing coverage for one observation.

    Args:
        sample: One observation of the world.
        profiles: Combat profiles by type name.
        extractor_family: Type names counted as extractors, every tier --
            the same family :func:`~rw_bot.policy.economy.count_extractors`
            counts, passed in so the two can never drift apart silently.

    Returns:
        The three coverage counts.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not
            describe a visible type.
    """
    hostiles = [entity for entity in sample["entities"] if entity["hostile"]]
    owned = [entity for entity in sample["entities"] if entity["mine"] and entity["complete"]]
    eco_covered = 0
    own_covered = 0
    for target in owned:
        if guns_covering(sample, profiles, target) == 0:
            continue
        own_covered += 1
        if target["type_name"] in extractor_family:
            eco_covered += 1
    foe_covered = 0
    for hostile in hostiles:
        for gun in owned:
            if not can_engage(profiles, gun, hostile):
                continue
            reach = reach_of(profiles, gun)
            if reach <= 0.0:
                continue
            span = (gun["x"] - hostile["x"]) ** 2 + (gun["y"] - hostile["y"]) ** 2
            if span <= reach * reach:
                foe_covered += 1
                break
    return Coverage(eco_covered=eco_covered, own_covered=own_covered, foe_covered=foe_covered)


__all__ = ["Coverage", "coverage", "guns_covering"]
