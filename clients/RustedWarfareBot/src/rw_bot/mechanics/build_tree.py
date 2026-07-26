"""Which type can make which, read from the engine's own registry.

The world stream already says what the player's units can make right now
([[wire-contract-ndjson]]). That is the right source for dispatch, because it
carries the engine id an order is addressed to and because availability is
genuinely per unit. It cannot answer the question a *plan* asks, though: a plan
reasons about things that do not exist yet, and nothing the player owns can make
a tank until a factory is standing.

So this decodes the static half — every producer-to-product edge in the
registry, dumped by ``make type-flags`` alongside the placement rules. Both
kinds ride in one file because they are two questions about the same types,
taken in one pass; two files could be regenerated against different game builds
and silently disagree.

The two sources cross-check, which is what makes either trustworthy. The
registry says a Builder makes thirteen structures; the live option stream,
reached by a completely different route, reports the same thirteen for the
Builder the player owns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from rw_bot import RwBotError
from rw_bot.validation import require_int, require_non_empty_str
from rw_bot.wire.ndjson import parse_object

KIND_BUILD_EDGE: Final = "buildedge"
"""``kind`` value of a build-tree record."""

KIND_UNIT_TYPE: Final = "unittype"
"""``kind`` value of a placement record, which this decoder skips."""

KIND_UNIT_COMBAT: Final = "unitcombat"
"""``kind`` value of a combat-stat record, which this decoder skips."""

_UNKNOWN_KIND = "RW-BUILDTREE-001"


class BuildTreeError(RwBotError):
    """The dump did not match the shape the agent writes.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description.
    """


def decode_build_tree(lines: Sequence[str]) -> dict[str, frozenset[str]]:
    """Decode the producer-to-product edges in a type dump.

    Placement records in the same file are skipped rather than rejected: one
    dump carries both kinds, and each decoder projects the one it needs. A kind
    that is neither is still an error, so a genuinely unknown record cannot pass
    silently through both readers.

    Args:
        lines: NDJSON lines, without newline terminators.

    Returns:
        Product type names by producer type name. A producer that makes nothing
        is absent rather than mapped to an empty set.

    Raises:
        NdjsonError: When a line does not parse.
        DecodeError: When a record is missing a field or carries a wrong type.
        BuildTreeError: ``RW-BUILDTREE-001`` on an unknown ``kind``.
    """
    edges: dict[str, set[str]] = {}
    for line in lines:
        if line.strip() == "":
            continue
        record = parse_object(line)
        kind = require_non_empty_str(record, "kind")
        if kind in (KIND_UNIT_TYPE, KIND_UNIT_COMBAT):
            # One dump, several kinds: each decoder projects its own and steps
            # over its neighbours'. A kind no decoder claims is still an error
            # in all of them, so nothing genuinely unknown passes silently --
            # but a kind added for one reader must be listed here, and this
            # decoder learned that by failing on a live run when a third kind
            # appeared ([[mechanics-build-tree]]).
            continue
        if kind != KIND_BUILD_EDGE:
            raise BuildTreeError(_UNKNOWN_KIND, f"unknown record kind {kind!r}")

        require_int(record, "index")
        producer = require_non_empty_str(record, "producer")
        produces = require_non_empty_str(record, "produces")
        edges.setdefault(producer, set()).add(produces)
    return {producer: frozenset(products) for producer, products in edges.items()}


def producers_of(tree: Mapping[str, frozenset[str]], product: str) -> frozenset[str]:
    """Return every type that can make ``product``.

    The inverse of the dumped direction, computed rather than stored. The dump
    is small — a few hundred edges — and holding one direction on disk keeps the
    artifact a plain list of facts instead of two views that could disagree.

    Args:
        tree: Products by producer.
        product: The type wanted.

    Returns:
        The producers, empty when nothing makes it.
    """
    return frozenset(producer for producer, products in tree.items() if product in products)


__all__ = [
    "KIND_BUILD_EDGE",
    "KIND_UNIT_COMBAT",
    "KIND_UNIT_TYPE",
    "BuildTreeError",
    "decode_build_tree",
    "producers_of",
]
