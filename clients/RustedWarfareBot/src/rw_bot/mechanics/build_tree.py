"""Which type can make which, read from the engine's own registry.

The world stream already says what the player's units can make right now
([[wire-contract-ndjson]]). That is the right source for dispatch, because it
carries the engine id an order is addressed to and because availability is
genuinely per unit. It cannot answer the question a *plan* asks, though: a plan
reasons about things that do not exist yet, and nothing the player owns can make
a tank until a factory is standing.

So this decodes the static half — every producer-to-product edge in the
registry, dumped by ``make type-flags`` alongside the placement rules and the
combat profiles. All three kinds ride in one file because they are three
questions about the same types, taken in one pass; separate files could be
regenerated against different game builds and silently disagree
([[mechanics-build-tree]]).

The two sources cross-check, which is what makes either trustworthy. The
registry says a Builder makes thirteen structures; the live option stream,
reached by a completely different route, reports the same thirteen for the
Builder the player owns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.registry_dump import KIND_BUILD_EDGE, records_of_kind
from rw_bot.validation import require_int, require_non_empty_str


def decode_build_tree(lines: Sequence[str]) -> dict[str, frozenset[str]]:
    """Decode the producer-to-product edges in a type dump.

    The dump's other kinds are stepped over by
    :func:`~rw_bot.mechanics.registry_dump.records_of_kind`, which is also what
    rejects a kind no decoder claims. This module used to carry that list
    itself and failed on a live run when a third kind was added for a different
    reader ([[mechanics-build-tree]]).

    Args:
        lines: NDJSON lines, without newline terminators.

    Returns:
        Product type names by producer type name. A producer that makes nothing
        is absent rather than mapped to an empty set.

    Raises:
        NdjsonError: When a line does not parse.
        DecodeError: When a record is missing a field or carries a wrong type.
        RegistryDumpError: ``RW-REGISTRY-001`` on a record kind the dump does
            not define.
    """
    edges: dict[str, set[str]] = {}
    for record in records_of_kind(lines, KIND_BUILD_EDGE):
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


def encode_build_edge(index: int, producer: str, produces: str) -> str:
    """Render one producer-to-product edge back to its NDJSON record.

    Round-trips with :func:`decode_build_tree`. The edge is passed as its three
    fields rather than as a TypedDict because the decoder folds edges into a
    mapping and never materialises one: inventing a record type so the encoder
    could take it would add a shape nothing else uses.

    Args:
        index: Position in the dump.
        producer: Type name that can make it.
        produces: Type name made.

    Returns:
        One NDJSON line, without a newline terminator.
    """
    return (
        f'{{"kind":"{KIND_BUILD_EDGE}","index":{index},'
        f'"producer":"{producer}","produces":"{produces}"}}'
    )


__all__ = [
    "decode_build_tree",
    "encode_build_edge",
    "producers_of",
]
