"""Things a suite needs exclusively, fleet-wide or one copy per node.

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
        JSONTypeError: If the value is not a list of non-empty strings, or
            a name holds :data:`NODE_SCOPE_SEPARATOR`. Empty is refused
            because a resource named ``""`` would be contended by every
            other empty name, silently serialising projects that share
            nothing.
    """
    names = _decode_strings(value, field=field)
    for index, entry in enumerate(names):
        if NODE_SCOPE_SEPARATOR in entry:
            raise JSONTypeError(
                f"{field}[{index}] is {entry!r}; {NODE_SCOPE_SEPARATOR!r} is reserved for the "
                "node a node-local resource's lease names, and a declared name holding it could "
                "contend as though it were one node's copy"
            )
    return names


def decode_held_names(value: JSONValue, *, field: str) -> tuple[str, ...]:
    """Decode the resource names a lease record holds.

    The declared names, as :func:`decode_names` admits them, except that a
    node-local one is written ``<name>@<node>`` (:func:`scoped`).

    Args:
        value: The value under ``field``, or None when the key was absent.
        field: The key it came from, for the message.

    Returns:
        The names, in record order, deduplicated.

    Raises:
        JSONTypeError: If the value is not a list of non-empty strings, or a
            name holds more than one :data:`NODE_SCOPE_SEPARATOR` or leaves
            either side of it empty, which no :func:`scoped` call writes.
    """
    names = _decode_strings(value, field=field)
    for index, entry in enumerate(names):
        if NODE_SCOPE_SEPARATOR not in entry:
            continue
        parts = entry.split(NODE_SCOPE_SEPARATOR)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise JSONTypeError(
                f"{field}[{index}] is {entry!r}; a node-local resource is recorded as "
                f"<name>{NODE_SCOPE_SEPARATOR}<node>, one separator and both sides named"
            )
    return names


def _decode_strings(value: JSONValue, *, field: str) -> tuple[str, ...]:
    """Decode a list of non-empty strings, deduplicated in order.

    Args:
        value: The value under ``field``, or None when the key was absent.
        field: The key it came from, for the message.

    Returns:
        The strings, empty for an absent list.

    Raises:
        JSONTypeError: If the value is not a list of non-empty strings.
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


#: Joins a node-local resource's name to the node whose copy a lease holds.
NODE_SCOPE_SEPARATOR = "@"


def scoped(names: tuple[str, ...], *, node: str, node_local: tuple[str, ...]) -> tuple[str, ...]:
    """Name the resources a dispatch to one node holds, each at its true scope.

    A FLEET-WIDE resource keeps its bare name, so it contends with every
    other holder anywhere. A NODE-LOCAL one, a thing each node runs its own
    copy of, is recorded as ``<name>@<node>``, so it contends only with a
    lease on the same node and :func:`contended` needs no second rule.

    THE CASE THAT MADE THE DISTINCTION. ``corvis-fleet-testdb`` was declared
    exclusive when diphtheria was its only host. It is a container each
    testdb node runs for itself (MCPs ``scripts/host/lib/fleet-testdb.sh``),
    and on 2026-09-29, the day lavender-wsl became the second such node
    (MCPs board task c4fc4f3e), every testdb job lavender-wsl claimed while
    diphtheria ran one was closed ``RESOURCE_HELD ... held fleet-wide``. Two
    testdb nodes still ran one testdb suite at a time.

    Args:
        names: The resources a project declares.
        node: The workspace name of the node the dispatch goes to.
        node_local: The workspace's node-local resource names.

    Returns:
        The names in declaration order, the node-local ones scoped to ``node``.
    """
    return tuple(
        f"{name}{NODE_SCOPE_SEPARATOR}{node}" if name in node_local else name for name in names
    )


def fleet_wide(names: tuple[str, ...], *, node_local: tuple[str, ...]) -> tuple[str, ...]:
    """The resources that are one thing across the whole fleet.

    What a check made before any node is chosen may ask about: a node-local
    resource held on one node says nothing about the copy on another.

    Args:
        names: The resources a project declares.
        node_local: The workspace's node-local resource names.

    Returns:
        ``names`` without the node-local ones, in declaration order.
    """
    return tuple(name for name in names if name not in node_local)


#: Begins the name a lease holds a companion directory under.
COMPANION_PREFIX = "companion:"


def companion_names(directories: tuple[str, ...], *, node: str) -> tuple[str, ...]:
    """Name the companion directories a dispatch to one node holds while it runs.

    A COMPANION IS ONE DIRECTORY PER NODE, REPLACED BY EVERY RUN THAT STAGES
    IT (:func:`fleet.core.names.companion_directory`), so two runs on one node
    that stage the same companion cannot overlap: the second one's staging
    deletes and rewrites the tree the first one's check is reading. On loki,
    NixOS 26.05, on 2026-10-26 (MCPs board task c2e3f31f), corvis-stick's run
    began at 22:36:18Z and Dashboards' at 22:36:24Z, both staging the MCPs
    companion, and corvis-stick's check, from 22:37:02Z to 22:37:37Z, failed
    lint and twelve cases with ``fatal: not a git repository:
    '/home/corvis/fleet/stage/MCPs/.git'`` while Dashboards rewrote it (fleet
    job ef321996). So each directory is held as that node's own copy, as a
    node-local resource is, and is contended only by a run on the same node.

    Args:
        directories: The companions' declared directory names.
        node: The workspace name of the node the dispatch goes to.

    Returns:
        ``companion:<directory>@<node>`` for each, in declaration order.
    """
    return tuple(
        f"{COMPANION_PREFIX}{directory}{NODE_SCOPE_SEPARATOR}{node}" for directory in directories
    )


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


__all__ = [
    "COMPANION_PREFIX",
    "NODE_SCOPE_SEPARATOR",
    "companion_names",
    "contended",
    "decode_held_names",
    "decode_names",
    "encode_names",
    "fleet_wide",
    "scoped",
]
