"""What a node CAN run, as tags derived from what was measured about it.

A project says what it needs (``required_tags``); a node never declares tags
at all. Its tags are DERIVED from the fields the node contract already
carries: its ``platform``, whether ``gpu`` is a device rather than None, and
whether it runs the fleet test database (``test_database``). That is the
whole reason this module exists beside :mod:`node`: a declared ``tags``
column on the node would be a second copy of those facts, and the copy is
the one that drifts (a box whose card was pulled would keep its ``gpu`` tag
until somebody remembered the list). Deriving means the tag is exactly as
true as the declaration it comes from.

WHY THESE FOUR. ``windows`` and ``linux`` because a suite may run on one
dialect only: slime's browser project launches Chromium with ANGLE over
Direct3D 11 and refuses every other platform by name
(``slime/scripts/chromium-launch.ts``, MCPs board task 41f45bd7), so it needs
``windows``, while every poetry project here runs under either dialect and
names neither. ``gpu`` because that same suite drives a real scene on the
machine's GPU and is too slow to run under the software rasteriser; the tag
means a CUDA device ``nvidia-smi`` reports, which is what ``gpu`` on the
node contract has always meant, and NOT an integrated adapter. The identity
registry (``fleet-mcp/fleet-nodes.json``) records both kinds per node with
the probe that measured each, so which windows nodes may carry this tag is a
recorded fact rather than a guess: on 2026-09-21 austinpc, sedona and
lavender, and diphtheria on linux. ``testdb`` because 17 MCPs packages start
their suites from ``packages/db``'s global test setup, which needs a migrated
``corvis_test`` and its owner role, and no Windows node can reach a test
database (MCPs board task 6bbfd171, measured 2026-09-26): the tag means the
node runs ``corvis-fleet-testdb``, the loopback container MCPs
``scripts/testdb-setup.sh`` restarts empty and migrates before each run, so
such a package lands only where its global setup can succeed.

A project naming both platforms is refused at decode: no node is both, so
the declaration could never match anything, and the honest way to say "either"
is to name neither.
"""

from __future__ import annotations

from typing import Final, Literal

from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.node import NodeConfig

#: A capability a project may require of a node.
NodeTag = Literal["windows", "linux", "gpu", "testdb"]

#: Every tag, for the decoder's refusal and the reader's reference. The
#: dispatch queue's vocabulary CHECK (MCPs migrations 532 and 563) is the
#: same four words.
NODE_TAGS: Final[tuple[NodeTag, ...]] = ("windows", "linux", "gpu", "testdb")


def node_tags(node: NodeConfig) -> frozenset[NodeTag]:
    """The tags a node carries, derived from its declaration.

    Args:
        node: The node's declaration.

    Returns:
        Its platform, plus ``gpu`` when the node declares a CUDA device and
        ``testdb`` when it declares the fleet test database.
    """
    tags: set[NodeTag] = {node["platform"]}
    if node["gpu"] is not None:
        tags.add("gpu")
    if node["test_database"]:
        tags.add("testdb")
    return frozenset(tags)


def decode_node_tag(value: JSONValue, *, field: str) -> NodeTag:
    """Read one tag into the closed set.

    Args:
        value: The declared value.
        field: The key it came from, for the message.

    Returns:
        The tag.

    Raises:
        JSONTypeError: If it is not a string naming one of :data:`NODE_TAGS`.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"{field} must be a string, got {type(value).__name__}")
    for tag in NODE_TAGS:
        if value == tag:
            return tag
    raise JSONTypeError(
        f"{field} must be one of {', '.join(NODE_TAGS)}, got {value!r}; a tag names a fact "
        "the node contract carries (its platform, a CUDA device nvidia-smi reports, or the "
        "fleet test database), and one it does not carry could never be satisfied"
    )


def decode_required_tags(value: JSONValue, *, field: str) -> tuple[NodeTag, ...]:
    """Decode a project's required tags.

    Args:
        value: The value under ``field``. REQUIRED: a project declares what
            it needs even when that is nothing, because an absent key on a
            project that needs a GPU would dispatch its suite to a box with
            no card and let it time out there.
        field: The key it came from, for the message.

    Returns:
        The tags, in declaration order. Empty means any reachable node.

    Raises:
        JSONTypeError: If the key is absent, the value is not a list, an
            entry is not a tag, a tag repeats, or both platforms are named.
    """
    if value is None:
        raise JSONTypeError(
            f"{field} is required: [] for a suite any node may run, else the tags it needs "
            f"from {', '.join(NODE_TAGS)}"
        )
    if not isinstance(value, list):
        raise JSONTypeError(f"{field} must be a list of tags, got {type(value).__name__}")
    tags: list[NodeTag] = []
    for index, entry in enumerate(value):
        tag = decode_node_tag(entry, field=f"{field}[{index}]")
        if tag in tags:
            raise JSONTypeError(f"{field}[{index}] repeats {tag!r}")
        tags.append(tag)
    if "windows" in tags and "linux" in tags:
        raise JSONTypeError(
            f"{field} names both windows and linux; no node is both, so nothing could satisfy "
            "it. Name neither to run on either platform"
        )
    return tuple(tags)


def encode_tags(tags: tuple[NodeTag, ...]) -> list[JSONValue]:
    """Encode tags for the workspace file.

    Args:
        tags: The tags to encode.

    Returns:
        The tags as a JSON list, in order.
    """
    return list(tags)


def missing_tags(node: NodeConfig, required: tuple[NodeTag, ...]) -> tuple[NodeTag, ...]:
    """The tags a project requires that a node does not carry.

    Args:
        node: The node's declaration.
        required: The project's required tags.

    Returns:
        The missing tags in the project's declaration order; empty when the
        node satisfies every requirement.
    """
    carried = node_tags(node)
    return tuple(tag for tag in required if tag not in carried)


__all__ = [
    "NODE_TAGS",
    "NodeTag",
    "decode_node_tag",
    "decode_required_tags",
    "encode_tags",
    "missing_tags",
    "node_tags",
]
