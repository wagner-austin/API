"""What a unit type can shoot, and which layers its fire reaches.

The bot used to ask one question here — how far can this shoot — and it was
half the question. ``c_tank``, the only unit the opening plan builds, declares
``canAttackFlyingUnits: false`` and ``canAttackUnderwaterUnits: false`` in its
own ``.ini``. Nothing in the planner read that. Combat filtered units on
*having* a weapon and never on the weapon *reaching the target*, so on a water
map the army could commit to a helicopter, hold that target for as long as it
stayed visible, and never fire a shot ([[policy-combat]]).

**The test here is the engine's, not a model of it.** The engine answers "can
this shoot that" in one method, and it branches three ways: a flying target is
tested against the attacker's air predicate, a submerged one against its
underwater predicate, and everything else against its ground predicate, with one
refinement for weapons that only reach targets standing in water.
:func:`can_engage` is that branch, transcribed. The attacker's four predicates
ride on this record; the target's three states ride on the entity
([[wire-contract-ndjson]]), so neither side is inferred.

Reach lives here too, rather than in the stat catalogue, and the coverage is
why. ``-printunits`` emits 90 of the engine's 173 registered types — it skips the
bug faction by name prefix, shadowed built-ins, types without a listing flag, and
sixteen names it blocklists — so a threat model reading it treats 48 armed types
as harmless, among them every turret ([[policy-threat]]). This reads the
registry, so every type answers. Where the two overlap they agree exactly, on all
90, which makes this a wider reading of the same fact rather than a second
opinion ([[mechanics-unit-catalogue]]).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot import RwBotError
from rw_bot.mechanics.registry_dump import KIND_UNIT_COMBAT, records_of_kind
from rw_bot.validation import (
    require_bool,
    require_finite_float,
    require_int,
    require_non_empty_str,
)
from rw_bot.wire.state import Entity

_DUPLICATE_TYPE = "RW-COMBAT-001"
_UNKNOWN_TYPE = "RW-COMBAT-002"


class CombatProfileError(RwBotError):
    """The combat dump did not describe the types it was asked about.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description, naming the type.
    """


class CombatProfile(TypedDict):
    """One type's reach, and the layers that reach covers.

    Attributes:
        index: Position in the dump. Enumeration order only.
        type_name: Engine type name, e.g. ``"c_tank"``. Joins to the ``type``
            field of a live entity.
        attack_range: Range in world units, comparable with entity positions.
            Zero for the unarmed.
        hits_land: Whether its fire reaches a target that is neither airborne
            nor submerged. False for the unarmed.
        hits_air: Whether its fire reaches an airborne target.
        hits_underwater: Whether its fire reaches a submerged target.
        hits_land_out_of_water: Whether its ground fire reaches a target clear
            of water. False only for weapons that must strike something in the
            water — a torpedo — and true for everything the base game lets a
            player build.
    """

    index: int
    type_name: str
    attack_range: float
    hits_land: bool
    hits_air: bool
    hits_underwater: bool
    hits_land_out_of_water: bool


def decode_combat_profiles(lines: Sequence[str]) -> Mapping[str, CombatProfile]:
    """Decode every combat record in a type-registry dump.

    Args:
        lines: NDJSON lines, without newline terminators.

    Returns:
        Combat profiles by type name.

    Raises:
        NdjsonError: When a line does not parse.
        DecodeError: When a record is missing a field or carries a wrong type.
        RegistryDumpError: ``RW-REGISTRY-001`` on a record kind the dump does
            not define.
        CombatProfileError: ``RW-COMBAT-001`` on a repeated type name.
    """
    profiles: dict[str, CombatProfile] = {}
    for record in records_of_kind(lines, KIND_UNIT_COMBAT):
        type_name = require_non_empty_str(record, "name")
        if type_name in profiles:
            raise CombatProfileError(
                _DUPLICATE_TYPE,
                f"type name {type_name!r} appears twice; it is the join key to live "
                "entities and must identify exactly one type",
            )
        profiles[type_name] = CombatProfile(
            index=require_int(record, "index"),
            type_name=type_name,
            attack_range=require_finite_float(record, "attack_range"),
            hits_land=require_bool(record, "hits_land"),
            hits_air=require_bool(record, "hits_air"),
            hits_underwater=require_bool(record, "hits_underwater"),
            hits_land_out_of_water=require_bool(record, "hits_land_out_of_water"),
        )
    return profiles


def encode_combat_profile(profile: CombatProfile) -> str:
    """Render a combat profile back to its NDJSON record.

    Round-trips with :func:`decode_combat_profiles`, which is what lets a
    decoded dump be re-emitted as a fixture rather than hand-written.

    Args:
        profile: The profile to encode.

    Returns:
        One NDJSON line, without a newline terminator.
    """
    return (
        f'{{"kind":"{KIND_UNIT_COMBAT}","index":{profile["index"]},'
        f'"name":"{profile["type_name"]}",'
        f'"attack_range":{profile["attack_range"]!r},'
        f'"hits_land":{str(profile["hits_land"]).lower()},'
        f'"hits_air":{str(profile["hits_air"]).lower()},'
        f'"hits_underwater":{str(profile["hits_underwater"]).lower()},'
        f'"hits_land_out_of_water":{str(profile["hits_land_out_of_water"]).lower()}}}'
    )


def profile_of(profiles: Mapping[str, CombatProfile], type_name: str) -> CombatProfile:
    """Return one type's combat profile.

    Indexed rather than looked up with a default, and that is deliberate. The
    dump is written from the registry and covers **every** registered type, so a
    missing name is a stale dump against a running game, not a gap to absorb —
    and absorbing it is what previously reported every turret as harmless
    ([[mechanics-movement-layers]] for the sibling case).

    Args:
        profiles: Combat profiles by type name.
        type_name: The type to read.

    Returns:
        Its profile.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            the type, naming it.
    """
    profile = profiles.get(type_name)
    if profile is None:
        raise CombatProfileError(
            _UNKNOWN_TYPE,
            f"the combat dump does not describe {type_name!r}; it is regenerated from "
            "the registry and covers every registered type, so this dump is stale "
            "against the running game",
        )
    return profile


def reach_of(profiles: Mapping[str, CombatProfile], entity: Entity) -> float:
    """Return how far an entity can shoot.

    Args:
        profiles: Combat profiles by type name.
        entity: The entity to measure.

    Returns:
        Attack range in world units, comparable with entity positions. Zero for
        a type the engine reports as unarmed.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            the entity's type.
    """
    return profile_of(profiles, entity["type_name"])["attack_range"]


def is_armed(profiles: Mapping[str, CombatProfile], entity: Entity) -> bool:
    """Report whether an entity has a weapon at all.

    Read from the registry rather than from the stat catalogue, because the
    catalogue describes 90 of 173 types and an absent entry there is
    indistinguishable from an unarmed unit. Here the two are distinct: every
    type has a record, and an unarmed one carries zero reach.

    Args:
        profiles: Combat profiles by type name.
        entity: The entity to test.

    Returns:
        True when the engine gives its type a weapon.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            the entity's type.
    """
    return reach_of(profiles, entity) > 0.0


def can_engage(profiles: Mapping[str, CombatProfile], attacker: Entity, target: Entity) -> bool:
    """Report whether the attacker's weapon reaches the target's layer.

    The engine's own test, transcribed branch for branch. Order matters and is
    the engine's: airborne is checked before submerged, and both before ground,
    so a unit that is somehow both is judged as the engine would judge it rather
    than as this function would prefer.

    Range is deliberately **not** part of the answer. Whether a target is
    reachable is a question about the weapon; whether it is in range right now
    is a question about the walk, and the caller that orders an attack expects
    the unit to close the distance ([[policy-combat]]).

    Args:
        profiles: Combat profiles by type name.
        attacker: The entity that would fire.
        target: The entity it would fire at.

    Returns:
        True when the attacker's weapon can reach the layer the target is on.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            either type.
    """
    profile = profile_of(profiles, attacker["type_name"])
    if target["flying"]:
        return profile["hits_air"]
    if target["submerged"]:
        return profile["hits_underwater"]
    if not profile["hits_land_out_of_water"] and not target["touching_water"]:
        return False
    return profile["hits_land"]


__all__ = [
    "CombatProfile",
    "CombatProfileError",
    "can_engage",
    "decode_combat_profiles",
    "encode_combat_profile",
    "is_armed",
    "profile_of",
    "reach_of",
]
