"""The type-registry dump, and the one place its record kinds are declared.

``make type-flags`` writes one file describing every registered unit type, and
that file carries several kinds of record because they are several questions
about the same types answered in one pass over one registry. Splitting them
across files would let two of them be regenerated against different game builds
and disagree silently, which is the failure the single dump exists to prevent.

Each kind has exactly one decoder: :mod:`rw_bot.mechanics.placement` claims
``unittype``, :mod:`rw_bot.mechanics.build_tree` claims ``buildedge``, and
:mod:`rw_bot.mechanics.combat_profile` claims ``unitcombat``. What every decoder
also needs is to step over its neighbours' records without accepting a record
nobody claims, and **that** is what lives here.

It lives here because the alternative was tried and drifted. Each decoder used
to carry its own hard-coded list of the other kinds, so adding a fourth kind
meant editing three unrelated modules, and the one time a kind was added the
build-tree decoder failed on a live run for exactly that reason
([[mechanics-build-tree]]). Declaring the kinds once and deriving "not mine"
from that makes a new kind a one-line change that cannot break a decoder that
does not care about it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from rw_bot import RwBotError
from rw_bot.validation import require_non_empty_str
from rw_bot.wire.ndjson import parse_object

KIND_UNIT_TYPE: Final = "unittype"
"""``kind`` value of a placement record: where a type may stand."""

KIND_BUILD_EDGE: Final = "buildedge"
"""``kind`` value of a build-tree record: one producer-to-product edge."""

KIND_UNIT_COMBAT: Final = "unitcombat"
"""``kind`` value of a combat record: reach, and the layers it reaches onto."""

KINDS: Final = frozenset({KIND_UNIT_TYPE, KIND_BUILD_EDGE, KIND_UNIT_COMBAT})
"""Every kind the dump may carry.

A record whose kind is absent from this set is rejected by every decoder. That
is the property worth having: an unrecognised record cannot pass silently
through all three readers by each one assuming another handles it.
"""

_UNKNOWN_KIND = "RW-REGISTRY-001"
_UNCLAIMED_KIND = "RW-REGISTRY-002"


class RegistryDumpError(RwBotError):
    """The dump carried a record no decoder can account for.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description, naming the offending kind.
    """


def records_of_kind(
    lines: Sequence[str], kind: str
) -> tuple[Mapping[str, str | int | float | bool], ...]:
    """Return every record of one kind, stepping over the dump's other kinds.

    Blank lines are skipped, which is what a file ending in a newline produces.

    Args:
        lines: NDJSON lines, without newline terminators.
        kind: The kind to project. Must be one of :data:`KINDS`; asking for a
            kind the dump does not define is a programming error rather than a
            data error, and it fails as loudly as a malformed record would.

    Returns:
        The matching records, in dump order.

    Raises:
        NdjsonError: When a line does not parse.
        DecodeError: When a record carries no ``kind`` field.
        RegistryDumpError: ``RW-REGISTRY-001`` when a record's kind is not one
            this dump defines, ``RW-REGISTRY-002`` when the caller asks for a
            kind that is not one either.
    """
    if kind not in KINDS:
        raise RegistryDumpError(
            _UNCLAIMED_KIND,
            f"{kind!r} is not a kind the type-registry dump defines; it carries {sorted(KINDS)}",
        )
    claimed: list[Mapping[str, str | int | float | bool]] = []
    for line in lines:
        if line.strip() == "":
            continue
        record = parse_object(line)
        found = require_non_empty_str(record, "kind")
        if found not in KINDS:
            raise RegistryDumpError(
                _UNKNOWN_KIND,
                f"unknown record kind {found!r}; the type-registry dump carries "
                f"{sorted(KINDS)}, so this file was written by a different agent build",
            )
        if found == kind:
            claimed.append(record)
    return tuple(claimed)


__all__ = [
    "KINDS",
    "KIND_BUILD_EDGE",
    "KIND_UNIT_COMBAT",
    "KIND_UNIT_TYPE",
    "RegistryDumpError",
    "records_of_kind",
]
