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

from collections.abc import Sequence
from typing import TypedDict

from rw_bot import RwBotError
from rw_bot.mechanics.registry_dump import KIND_UNIT_TYPE, records_of_kind
from rw_bot.validation import require_bool, require_int, require_non_empty_str

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
        RegistryDumpError: ``RW-REGISTRY-001`` on a record kind the dump does
            not define.
        PlacementError: ``RW-PLACEMENT-002`` on a repeated type name.
    """
    placements: list[TypePlacement] = []
    seen: set[str] = set()

    for record in records_of_kind(lines, KIND_UNIT_TYPE):
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


def encode_placement(placement: TypePlacement) -> str:
    """Render a placement rule back to its NDJSON record.

    Round-trips with :func:`decode_placements`, which is what lets a decoded
    dump be re-emitted as a fixture rather than hand-written.

    Args:
        placement: The rule to encode.

    Returns:
        One NDJSON line, without a newline terminator.
    """
    return (
        f'{{"kind":"{KIND_UNIT_TYPE}","index":{placement["index"]},'
        f'"name":"{placement["type_name"]}",'
        f'"needs_pool":{str(placement["needs_pool"]).lower()}}}'
    )


__all__ = [
    "PlacementError",
    "TypePlacement",
    "decode_placements",
    "encode_placement",
]
