"""Things a suite needs exclusively that no single node owns.

WHY THE NODE BUDGET COULD NOT EXPRESS THIS. Every field in
:class:`~fleet.contracts.budget.NodeBudget` -- reserved cores, reserved and
per-worker memory, concurrent runs, disk -- describes a resource the NODE
owns, and the capacity check divides them. That is the right model for a
suite whose cost is CPU and RAM, which is every project registered here so
far, and it cannot describe a suite whose cost is a thing there is exactly ONE
of in the world.

THE CASE THAT FORCED IT, raised by ``opus-lavender-gpu-0824`` on 2026-09-04
and confirmed against ``packages/db/Makefile`` in the MCPs monorepo: its
``test`` target is ``migrate-test`` then vitest, and ``migrate-test`` applies
migrations to a single shared ``corvis_test`` database. Two sessions running
it locally already deadlock on an ``AccessExclusiveLock``. Distributing that
suite across nodes moves the CPU contention off one machine and leaves the
DATABASE contention exactly where it was -- and makes it worse, because two
nodes cannot see each other's processes at all, and a per-node capacity check
admits both while neither node is short of anything.

SO A LEASE, AND THE SAME LEASE. The primitive that already stops two
dispatches sharing one ``.venv`` is the right shape for this; it was simply
scoped one level too narrow, keyed on ``(node, project)``. A dispatch now also
names the fleet-wide resources it will hold, in the SAME lease record, and
:func:`contended` is what a second dispatch tests against. One lease per
dispatch, one file, one expiry, one release -- no second kind of claim to keep
in step with the first.

WHY THE NAMES ARE FREE STRINGS DECLARED IN THE WORKSPACE. What is exclusive
is a fact about the world, not about this package: ``corvis_test`` is one
database because there is one, and no amount of introspection here would
discover that. The workspace names it, every project that touches it names the
same string, and the fleet serialises them. An enum would mean this package
had to be edited before anybody could protect a new resource.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError, JSONValue


def decode_names(value: JSONValue, *, field: str) -> tuple[str, ...]:
    """Decode and validate a declared list of exclusive resource names.

    Args:
        value: The value under ``field``, or None when the key was absent.
        field: The key it came from, for the message.

    Returns:
        The names, in declaration order, deduplicated. An absent list decodes
        as empty: a project that needs no exclusive resource is the ordinary
        case and should not have to say so.

    Raises:
        JSONTypeError: If the value is not a list of non-empty strings.
            Empty is refused because a resource named ``""`` would be
            contended by every other empty name, silently serialising
            projects that share nothing.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise JSONTypeError(f"{field} must be a list of strings, got {type(value).__name__}")
    names: list[str] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, str):
            raise JSONTypeError(f"{field}[{index}] must be a string, got {type(entry).__name__}")
        if not entry.strip():
            raise JSONTypeError(
                f"{field}[{index}] is empty; a resource with no name is contended by every "
                "other unnamed one, which would serialise projects that share nothing"
            )
        if entry not in names:
            names.append(entry)
    return tuple(names)


def encode_names(resources: tuple[str, ...]) -> list[JSONValue]:
    """Encode a list of resource names.

    Args:
        resources: The names to encode.

    Returns:
        A JSON-serialisable list.
    """
    return list(resources)


def contended(held: tuple[str, ...], wanted: tuple[str, ...]) -> tuple[str, ...]:
    """Name the resources a would-be holder cannot have.

    Args:
        held: What an existing lease holds.
        wanted: What a new dispatch is asking for.

    Returns:
        The names in both, in ``wanted``'s order. Ordered by the ASKER's
        declaration rather than the holder's, because the message this
        produces is read by the asker and should list its own resources in
        the order it named them.
    """
    return tuple(name for name in wanted if name in held)


__all__ = ["contended", "decode_names", "encode_names"]
