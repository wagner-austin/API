"""Tilting the army mix toward what the opponent is actually fielding.

The composition is something the caller asks for, and until now it was asked
for blind: the loop records what the opponents field on every observation and
never read its own record. Three matches ended with 33 identical ``c_tank``
against air the type cannot shoot at, and the fix then was to allow a mix --
but the mix itself stayed static however the opponent played
([[mechanics-combat-profile]]).

This closes that loop without inventing a value model, by the same reasoning
production already follows: the shares are read from the world rather than
scored. The visible hostiles have a layer distribution, and the mix is tilted
until the share of it that can reach the air matches the share of the threat
that is airborne. No unit is invented -- only types the caller already asked
for are repeated, so a doctrine with no anti-air in it is left alone and the
gap stays visible where it already shows, in the report's ``engageable`` count.

The same arithmetic answers the water since 2026-08-07: the VH wall turned
out to be the opponents' naval branch, whose surface ships cannot be engaged
profitably by anything that stands inside their guns -- and seven one-knob
composition arms measured that permanently reshaping the mix for that case
costs the land game more than it pays ([[policy-exact-timing]], the naval
wall; log 2026-08-07, the arm ladder). So the answer is conditional, here:
when the seen threat includes WATER-layer movers, the tilt repeats the
doctrine's own types whose surface fire starts beyond the fleet's longest
reach. A mix with no such type is left alone, exactly like the airless case.

Pure, like the rest of the policy layer: entities in, a composition out.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import ceil
from typing import Final, TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile, profile_of
from rw_bot.policy.intel import Intel


class Threat(TypedDict):
    """What the tilt needs to know about one hostile: its type and its layer.

    Deliberately the narrowest reading. A live entity is a threat (width
    subtyping supplies the extra keys), and so is a remembered sighting
    (:class:`~rw_bot.policy.intel.Sighting`) -- which is what lets scouted
    intel and gun-range contact feed the same arithmetic
    ([[community-play-strategies]]).

    Attributes:
        type_name: Engine type name, for the profile lookup.
        flying: Whether it is (or was last seen) airborne.
        movement: The engine's movement layer for the type, e.g. ``"LAND"``
            or ``"WATER"`` -- the field a live entity already carries on the
            wire, and the one fact that tells a ship from everything else
            ([[policy-exact-timing]], the naval wall).
    """

    type_name: str
    flying: bool
    movement: str


#: The engine's movement-layer name for surface ships, as the wire spells it.
_NAVAL_LAYER: Final = "WATER"


def _hits_air(profiles: Mapping[str, CombatProfile], type_name: str) -> bool:
    """Report whether a type's weapon reaches an airborne target.

    Args:
        profiles: Combat profiles by type name.
        type_name: The type to test.

    Returns:
        True when its fire reaches the air.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            the type.
    """
    return profile_of(profiles, type_name)["hits_air"]


def _outguns(profiles: Mapping[str, CombatProfile], type_name: str, reach: float) -> bool:
    """Report whether a type engages a surface fleet from beyond its guns.

    The naval answer the profiles hand over mechanically: a weapon that hits
    the surface from further than the longest naval reach in sight fights
    the fleet without standing in its fire -- measured on the VH wall, where
    the 290-range artillery outranges every surface ship the opponents field
    ([[policy-exact-timing]], the naval wall).

    Args:
        profiles: Combat profiles by type name.
        type_name: The type to test.
        reach: The longest attack range among the naval threats in sight.

    Returns:
        True when its surface fire starts beyond that reach.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            the type.
    """
    record = profile_of(profiles, type_name)
    return record["hits_land"] and record["attack_range"] > reach


def _tilted(
    mix: tuple[str, ...],
    matched: int,
    total: int,
    answers: Callable[[str], bool],
    profiles: Mapping[str, CombatProfile],
) -> tuple[str, ...]:
    """Return the mix with enough answering entries to cover a threat share.

    The share arithmetic both tilts run on, extracted so the air and naval
    clauses cannot drift apart. The tilt only ever repeats types already in
    the mix, in their stated order, because repeats are how a composition
    says "more of this" ([[policy-production]]).

    Args:
        mix: The army mix to hold, repeats meaningful as a ratio.
        matched: How many visible threats this tilt answers.
        total: How many threats are visible in all.
        answers: Whether one mix type answers this tilt's threat.
        profiles: Combat profiles by type name, for the unarmed test in the
            everything-matches branch.

    Returns:
        The mix to produce against, repeats meaningful as a ratio.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            a type in the mix.
    """
    if matched == 0:
        return mix
    capable = tuple(name for name in dict.fromkeys(mix, True) if answers(name))
    if not capable:
        return mix
    if matched == total:
        return tuple(
            name
            for name in mix
            if answers(name) or profile_of(profiles, name)["attack_range"] == 0.0
        )
    share = matched / total
    held = sum(1 for name in mix if answers(name))
    # Append k answering entries so that (held + k) / (len(mix) + k) >= share.
    # Solved for k rather than looped, and the repeats cycle through every
    # capable type so a doctrine with two answers keeps its own ratio between
    # them.
    lack = share * len(mix) - held
    if lack <= 0:
        return mix
    wanted = ceil(lack / (1.0 - share))
    return (*mix, *(capable[i % len(capable)] for i in range(wanted)))


def counter_composition(
    composition: Sequence[str],
    targets: Sequence[Threat],
    profiles: Mapping[str, CombatProfile],
) -> tuple[str, ...]:
    """Return the mix, tilted until it answers the air and naval shares.

    Two passes of one arithmetic (:func:`_tilted`), air then naval, each
    following the same rule: only types already in the mix are repeated.
    Three cases fall out of that rule rather than being decided here:

    * **Nothing visible flies or floats**: the mix is returned unchanged.
      Fog is not evidence of absence, but neither is it evidence -- the tilt
      follows what is seen, exactly as targeting does.
    * **The mix has no answering type**: it is returned unchanged. Adding a
      type the caller never asked for would be a choice of unit made here,
      and which unit answers air -- or outranges a fleet -- is the
      doctrine's question, not this function's.
    * **Everything visible flies (or floats)**: the armed types that cannot
      answer are dropped for as long as that holds, because producing them
      is producing units that cannot fight anything in sight. Unarmed types
      stay -- a builder is in the mix for the economy, not the fight.

    The naval answer is range dominance: a type whose surface fire starts
    beyond the longest naval reach in sight fights the fleet without
    standing in its fire. Seven composition arms measured that FORCING a
    naval answer into the mix costs the land game more than it pays
    anywhere; this tilt spends nothing until a fleet is actually seen, and
    amplifies only what the doctrine already holds
    (log 2026-08-07, the arm ladder).

    Args:
        composition: The army mix to hold, repeats meaningful as a ratio.
        targets: The hostile entities currently visible.
        profiles: Combat profiles by type name, for reach and layers.

    Returns:
        The mix to produce against, repeats meaningful as a ratio.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            a type in the mix or a naval threat.
    """
    mix = tuple(composition)
    if not mix or not targets:
        return mix
    flying = sum(1 for target in targets if target["flying"])

    def hits_air(name: str) -> bool:
        return _hits_air(profiles, name)

    mix = _tilted(mix, flying, len(targets), hits_air, profiles)
    naval = tuple(t for t in targets if t["movement"] == _NAVAL_LAYER)
    if not naval:
        return mix
    reach = max(profile_of(profiles, t["type_name"])["attack_range"] for t in naval)

    def outguns(name: str) -> bool:
        return _outguns(profiles, name, reach)

    return _tilted(mix, len(naval), len(targets), outguns, profiles)


def mobile_threats(intel: Intel, catalogue: Mapping[str, UnitStats]) -> tuple[Threat, ...]:
    """Return the remembered sightings the counter tilt should read.

    **Mobile units only, and that was measured rather than assumed.** With
    scouting on, the tilt reads the remembered picture -- which includes
    everything currently visible, since the memory is fed each observation.
    V1 fed it everything remembered: a scouted base is mostly buildings and
    boats, so the flying share collapsed toward zero and the arm finished
    with less anti-air than the unscouted control. Seeing everything made the
    mix blinder than seeing only what attacked
    ([[community-play-strategies]]).

    Args:
        intel: The fog memory, already fed this observation.
        catalogue: Unit stats by type name, for the speed that tells a
            building from a unit. A type the catalogue does not price is
            dropped, which errs toward the unscouted behaviour.

    Returns:
        The remembered mobile hostiles, in identity order.
    """
    threats: list[Threat] = []
    for sighting in intel.remembered():
        stats = catalogue.get(sighting["type_name"])
        if stats is not None and stats["speed"] > 0.0:
            threats.append(sighting)
    return tuple(threats)


__all__ = ["Threat", "counter_composition", "mobile_threats"]
