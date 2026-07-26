"""Where the engine will let a structure stand, decoded from the agent's dump.

The stat catalogue answers what a unit costs and how fast it moves
([[mechanics-unit-catalogue]]). It does not answer where the unit may be put,
and one placement rule decides whether the bot has an economy at all: an
extractor may only be built on a resource pool, and the engine refuses it
silently anywhere else.

That rule is not written down anywhere readable. ``-printunits`` does not print
it. The only mention of it in the shipped files is an English sentence in a
translations bundle, which is a blurb rather than a fact — a bot that read it
would be parsing marketing copy. So the agent asks each unit type for its own
placement predicate and writes the answers out, and this module decodes them.

One flat object per type, so the same strict parsing the world stream uses
applies here unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

from rw_bot import RwBotError
from rw_bot.validation import (
    require_bool,
    require_finite_float,
    require_int,
    require_non_empty_str,
)
from rw_bot.wire.ndjson import parse_object

KIND_UNIT_TYPE: Final = "unittype"
"""``kind`` value of a placement record."""

KIND_BUILD_EDGE: Final = "buildedge"
"""``kind`` value of a build-tree record, which this decoder skips."""

KIND_UNIT_COMBAT: Final = "unitcombat"
"""``kind`` value of a combat record, decoded by :func:`decode_reaches`."""

_UNKNOWN_KIND = "RW-PLACEMENT-001"
_DUPLICATE_TYPE = "RW-PLACEMENT-002"


class PlacementError(RwBotError):
    """The placement dump did not match the shape the agent writes.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description, naming the type where known.
    """


class TypePlacement(TypedDict):
    """Where one unit type may be built.

    Attributes:
        index: Position in the dump. Enumeration order only.
        type_name: Engine type name, e.g. ``"extractorT1"``. The same string a
            live entity reports and the same one a build order carries.
        needs_pool: Whether the engine will only accept this type on a resource
            pool.
    """

    index: int
    type_name: str
    needs_pool: bool


def decode_placements(lines: Sequence[str]) -> tuple[TypePlacement, ...]:
    """Decode every record in a placement dump.

    Args:
        lines: NDJSON lines, without newline terminators.

    Returns:
        Every type, in dump order.

    Raises:
        NdjsonError: When a line does not parse.
        DecodeError: When a record is missing a field or carries a wrong type.
        PlacementError: ``RW-PLACEMENT-001`` on an unknown ``kind``,
            ``RW-PLACEMENT-002`` on a repeated type name.
    """
    placements: list[TypePlacement] = []
    seen: set[str] = set()

    for line in lines:
        if line.strip() == "":
            continue
        record = parse_object(line)
        kind = require_non_empty_str(record, "kind")
        if kind in (KIND_BUILD_EDGE, KIND_UNIT_COMBAT):
            # One dump carries both kinds -- where a type may stand, and what it
            # can make -- because they are one pass over one registry and two
            # files could be regenerated against different builds and disagree.
            # Each decoder projects its own kind; a kind neither claims is still
            # an error in both, so nothing unknown passes silently.
            continue
        if kind != KIND_UNIT_TYPE:
            raise PlacementError(_UNKNOWN_KIND, f"unknown record kind {kind!r}")

        type_name = require_non_empty_str(record, "name")
        if type_name in seen:
            raise PlacementError(
                _DUPLICATE_TYPE,
                f"type name {type_name!r} appears twice; the agent resolves shadowed "
                "names before writing, so a repeat means the dump is concatenated "
                "from two runs",
            )
        seen.add(type_name)
        placements.append(
            TypePlacement(
                index=require_int(record, "index"),
                type_name=type_name,
                needs_pool=require_bool(record, "needs_pool"),
            )
        )
    return tuple(placements)


def decode_reaches(lines: Sequence[str]) -> Mapping[str, float]:
    """Decode how far every registered type can shoot.

    Separate from the catalogue on purpose, and it is the coverage that makes
    the difference. ``-printunits`` emits 90 of the engine's 173 registered
    types — it skips the bug faction by name prefix, shadowed built-ins, types
    without a listing flag, and sixteen names it blocklists outright — so a
    threat model reading it treats 48 armed types as harmless, among them every
    turret and the artillery ([[policy-threat]]).

    This asks the registry instead, so every type answers. Where the two
    overlap they agree exactly, on all 90, which is what makes this a wider
    reading of the same fact rather than a second opinion
    ([[mechanics-unit-catalogue]]).

    Args:
        lines: NDJSON lines, without newline terminators.

    Returns:
        Attack range in world units by type name, zero for the unarmed.

    Raises:
        NdjsonError: When a line does not parse.
        DecodeError: When a record is missing a field or carries a wrong type.
        PlacementError: ``RW-PLACEMENT-001`` on an unknown ``kind``,
            ``RW-PLACEMENT-002`` on a repeated type name.
    """
    reaches: dict[str, float] = {}
    for line in lines:
        if line.strip() == "":
            continue
        record = parse_object(line)
        kind = require_non_empty_str(record, "kind")
        if kind in (KIND_BUILD_EDGE, KIND_UNIT_TYPE):
            continue
        if kind != KIND_UNIT_COMBAT:
            raise PlacementError(_UNKNOWN_KIND, f"unknown record kind {kind!r}")
        type_name = require_non_empty_str(record, "name")
        if type_name in reaches:
            raise PlacementError(
                _DUPLICATE_TYPE,
                f"type name {type_name!r} appears twice; it is the join key to live "
                "entities and must identify exactly one type",
            )
        reaches[type_name] = require_finite_float(record, "attack_range")
    return reaches


__all__ = [
    "KIND_BUILD_EDGE",
    "KIND_UNIT_COMBAT",
    "KIND_UNIT_TYPE",
    "PlacementError",
    "TypePlacement",
    "decode_placements",
    "decode_reaches",
]
