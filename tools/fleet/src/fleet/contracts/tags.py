"""What a node CAN run, as tags derived from what was measured about it.

A project says what it needs (``required_tags``); a node never declares tags
at all. Its tags are DERIVED from the fields the node contract already
carries: its ``platform``, whether ``gpu`` is a device rather than None,
whether it runs the fleet test database (``test_database``), and which
toolchains it names by version (``rust``, ``cxx``). That is the
whole reason this module exists beside :mod:`node`: a declared ``tags``
column on the node would be a second copy of those facts, and the copy is
the one that drifts (a box whose card was pulled would keep its ``gpu`` tag
until somebody remembered the list). Deriving means the tag is exactly as
true as the declaration it comes from.

A RUNNER CLAIMS WITH WHAT ITS NODE ANSWERS, NOT WITH THIS (MCPs board task
939ec5c7). :func:`node_tags` is what the hub's own planning reads without
asking a node (``fleet-run``, preflight); the runner on each node re-reads
every one of these facts from the node every tick and claims with that
(:mod:`fleet.contracts.detection`), comparing it with the declaration and
logging any difference, so installing a tool makes a node eligible on its
next tick with no file edited.

WHY THESE SIX. ``windows`` and ``linux`` because a suite may run on one
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
``corvis_test`` and its owner role, and a node without its own container
cannot reach one (MCPs board task 6bbfd171, measured 2026-09-26): the tag
means the node runs ``corvis-fleet-testdb``, on Linux or under Docker Desktop
on Windows (sedona, MCPs board task daae17f2), the loopback container MCPs
``scripts/testdb-setup.sh`` restarts empty and migrates before each run, so
such a package lands only where its global setup can succeed. ``rust``
because API ``services/covenant-radar-api`` builds the maturin crate
``libs/cleargbm_rs`` from source in ``poetry sync``, and on 2026-09-26 no
node carried cargo (MCPs board task 1e2da299): the tag means the node
declares the version its cargo prints, which the runner's probe re-measures
every tick. ``cxx`` because every MCPs TypeScript project's root ``npm ci``
rebuilds ``hnswlib-node`` under node-gyp, and on 2026-09-27 no Windows node
had the VC tools it needs (MCPs board task 3f19c136): the tag means the node
declares the version of the C++ toolchain its probe finds. ``docker``
because MCPs/execution-deploy runs the real make deploy and, on the
operator's ruling of 2026-09-27, may only do so on a rootless daemon under a
user outside the docker group, never the stack's (MCPs board task
6c4516af): the tag means the node declares that daemon's version, which its
probe reads only when the daemon says it is rootless. ``stack`` because
three MCPs suites start the corvis compose stack's own images on its network
inside their ``make check``, and on 2026-09-29 lavender-wsl, the second
testdb node, failed them where ``docker run`` found neither (MCPs board task
554bffc1): the tag means the node declares the version of the daemon its own
account reaches, which its probe reads only while that daemon holds the
network and the images. All four are :mod:`fleet.contracts.capability`.
``ffmpeg`` because API ``services/grandma-api``'s check converts real audio
files through it, and it used to be a tool every build required, so a node
without it claimed nothing at all: pendragon logged ``NODE_TOOL_MISSING`` for
ffmpeg on every tick of 2026-10-02 until 02:48Z while tag-free jobs waited
(MCPs board task 939ec5c7). It is the one tag no declaration carries: a
runner adds it when this tick's toolchain probe found the executable
(:func:`tool_tags`), so installing ffmpeg makes the node eligible on its next
tick, and a node without it takes every other job.
``hooks`` because MCPs ``packages/claude-hooks``'s check runs its suites on
the system interpreter, as Claude Code runs the hooks, and its live cases
reach the board through the build account's ``~/.claude/corvis-hooks.json``.
Measured 2026-10-02 (MCPs board task ec895824): sedona's account carried that
route file, pendragon's and serendipity's did not, and none of the three
interpreters had the check's tools. It is a tool tag like ``ffmpeg``: the
probe's ``hooks`` line answers present only when the route file exists and
the interpreter imports every tool the check runs.
``go`` because MCPs ``rcs-bridge`` is a Go module and no fleet node had go on
2026-10-05, when the SMS gateway it feeds was found running ahead of the
server it reads and both host processes had to become fleet projects (MCPs
board task 1da15750). It is a tool tag like ``ffmpeg``: the probe's ``go``
line answers present when ``go`` is on the build account's PATH.
``chrome`` because MCPs ``web-scraper``'s check launches Playwright's
channel ``chrome``, the system Google Chrome, in three real-browser cases,
and diphtheria, a ``cxx`` node without it, failed exactly those three on
2026-10-04 (fleet job 874c160c) and again on 2026-10-09 (56d7fbb5) while
passing the other 490, so the suite's verdict depended on which node
claimed it (MCPs board task 2f596185). It is a tool tag like ``ffmpeg``,
with one difference in how it is found: Chrome is not on a PATH, so the
probe's ``chrome`` line answers present when the binary exists at the path
Playwright launches, ``/opt/google/chrome/chrome`` on Linux and
``Google\\Chrome\\Application\\chrome.exe`` under a Program Files root on
Windows (:mod:`fleet.contracts.tagged_tools`).
``elevated`` because MCPs' Task Scheduler
installers register what only an administrator may register, and every
build launches as an S4U task at RunLevel Limited (MCPs board task
a98d7083): the tag means the node declares ``elevated`` and runs a SECOND
runner whose builds launch at RunLevel Highest, and that runner re-measures
its ssh session's token every tick (:mod:`fleet.contracts.elevation`).

``elevated`` IS EXCLUSIVE. Every other tag is a floor, so a runner claims
whatever it can satisfy; a node's ordinary runner claims without
``elevated`` (:func:`runner_tags`), and the queue's claim gives an elevated
job only to a runner carrying the tag and that runner only elevated jobs
(MCPs ``FLEET_DISPATCH_EXCLUSIVE_TAGS``), so an ordinary suite never runs as
an administrator and an elevated one never waits behind ordinary checks.

A project naming both platforms is refused at decode: no node is both, so
the declaration could never match anything, and the honest way to say "either"
is to name neither.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.members import find_member

from fleet.contracts.capability import Capability
from fleet.contracts.node import NodeConfig, NodePlatform, declared_capability
from fleet.contracts.toolchain import ToolReport


class NodeTag(StrEnum):
    """A capability a project may require of a node.

    The dispatch queue's vocabulary CHECK (MCPs migrations 532, 563, 569,
    570, 571, 615, 622, 639, 648, 668 and 707) is the same thirteen words as
    these members' values, in this order.
    """

    WINDOWS = "windows"
    LINUX = "linux"
    GPU = "gpu"
    TESTDB = "testdb"
    RUST = "rust"
    CXX = "cxx"
    DOCKER = "docker"
    ELEVATED = "elevated"
    STACK = "stack"
    FFMPEG = "ffmpeg"
    HOOKS = "hooks"
    GO = "go"
    CHROME = "chrome"


#: The tag each platform carries. A table rather than a lookup by word, so
#: the two vocabularies stay separate types; test_tags pins that every
#: platform has a row, which is what a third platform would need first.
PLATFORM_TAG: Final[dict[NodePlatform, NodeTag]] = {
    NodePlatform.WINDOWS: NodeTag.WINDOWS,
    NodePlatform.LINUX: NodeTag.LINUX,
}

#: The tag each declared toolchain carries, for the same reason.
CAPABILITY_TAG: Final[dict[Capability, NodeTag]] = {
    Capability.RUST: NodeTag.RUST,
    Capability.CXX: NodeTag.CXX,
    Capability.DOCKER: NodeTag.DOCKER,
    Capability.STACK: NodeTag.STACK,
}

#: The tag each detected tool carries, keyed by the line the toolchain probe
#: reports it under (:class:`fleet.contracts.toolchain.ToolReport`): the
#: executable's name, ``hooks`` for the hooks check's environment, or
#: ``chrome`` for Google Chrome at the path Playwright launches. These are
#: the one kind of tag no declaration holds: a runner adds them when this
#: tick's probe found the tool, so installing it makes the node eligible on
#: its next tick and removing it takes the tag away on the next.
TOOL_TAG: Final[dict[str, NodeTag]] = {
    "ffmpeg": NodeTag.FFMPEG,
    "hooks": NodeTag.HOOKS,
    "go": NodeTag.GO,
    "chrome": NodeTag.CHROME,
}


def node_tags(node: NodeConfig) -> frozenset[NodeTag]:
    """The tags a node carries, derived from its declaration.

    Args:
        node: The node's declaration.

    Returns:
        Its platform, plus ``gpu`` when the node declares a CUDA device,
        ``testdb`` when it declares the fleet test database, ``elevated``
        when it declares an elevated runner, and ``rust``, ``cxx``,
        ``docker`` or ``stack`` for each capability it declares a version of.
    """
    tags: set[NodeTag] = {PLATFORM_TAG[node["platform"]]}
    if node["gpu"] is not None:
        tags.add(NodeTag.GPU)
    if node["test_database"]:
        tags.add(NodeTag.TESTDB)
    if node["elevated"]:
        tags.add(NodeTag.ELEVATED)
    tags.update(
        tag
        for capability, tag in CAPABILITY_TAG.items()
        if declared_capability(node, capability) is not None
    )
    return frozenset(tags)


def runner_tags(node: NodeConfig, *, elevated: bool) -> frozenset[NodeTag]:
    """The tags one of a node's runners claims with.

    Args:
        node: The node's declaration.
        elevated: Whether this is the node's elevated runner.

    Returns:
        :func:`node_tags` for the elevated runner, and the same without
        ``elevated`` for the ordinary one, so the queue's exclusive rule
        hands each runner only its own lane's jobs.

    Raises:
        ValueError: For an elevated runner of a node that declares none; the
            runner registration creates one only for a node that does, so
            this is a runner started by hand against the wrong node.
    """
    carried = node_tags(node)
    if not elevated:
        return carried - {NodeTag.ELEVATED}
    if NodeTag.ELEVATED not in carried:
        raise ValueError(
            f"{node['host']} declares no elevated runner, so an elevated runner may not claim "
            "for it; set elevated to true only for a Windows node whose ssh account is an "
            "administrator"
        )
    return carried


def tool_tags(reports: tuple[ToolReport, ...]) -> frozenset[NodeTag]:
    """The tags of the tools a toolchain probe found on a node.

    Args:
        reports: What the node answered this tick.

    Returns:
        :data:`TOOL_TAG`'s tag for every tool the probe reported present;
        empty when it found none of them.
    """
    return frozenset(
        TOOL_TAG[report["name"]]
        for report in reports
        if report["present"] and report["name"] in TOOL_TAG
    )


def decode_node_tag(value: JSONValue, *, field: str) -> NodeTag:
    """Read one tag into the closed set.

    Args:
        value: The declared value.
        field: The key it came from, for the message.

    Returns:
        The tag.

    Raises:
        JSONTypeError: If it is not a string naming one of :class:`NodeTag`'s
            words.
    """
    if not isinstance(value, str):
        raise JSONTypeError(f"{field} must be a string, got {type(value).__name__}")
    tag = find_member(value, NodeTag)
    if tag is not None:
        return tag
    raise JSONTypeError(
        f"{field} must be one of {', '.join(NodeTag)}, got {value!r}; a tag names a fact "
        "the node contract carries (its platform, a CUDA device nvidia-smi reports, the "
        "fleet test database, a Rust or C++ toolchain, the execution suite's rootless "
        "Docker daemon, an elevated runner, the corvis compose stack, or a tool, the "
        "hooks check's environment or Google Chrome its probe found), and one it does not "
        "carry could never be satisfied"
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
            f"from {', '.join(NodeTag)}"
        )
    if not isinstance(value, list):
        raise JSONTypeError(f"{field} must be a list of tags, got {type(value).__name__}")
    tags: list[NodeTag] = []
    for index, entry in enumerate(value):
        tag = decode_node_tag(entry, field=f"{field}[{index}]")
        if tag in tags:
            raise JSONTypeError(f"{field}[{index}] repeats {tag.value!r}")
        tags.append(tag)
    if NodeTag.WINDOWS in tags and NodeTag.LINUX in tags:
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
    return [tag.value for tag in tags]


def missing_tags(carried: frozenset[NodeTag], required: tuple[NodeTag, ...]) -> tuple[NodeTag, ...]:
    """The tags a project requires that a runner does not carry.

    Args:
        carried: The tags the runner carries: :func:`node_tags`, or a
            runner's claim tags with :func:`tool_tags` added.
        required: The project's required tags.

    Returns:
        The missing tags in the project's declaration order; empty when the
        runner satisfies every requirement.
    """
    return tuple(tag for tag in required if tag not in carried)


__all__ = [
    "CAPABILITY_TAG",
    "PLATFORM_TAG",
    "TOOL_TAG",
    "NodeTag",
    "decode_node_tag",
    "decode_required_tags",
    "encode_tags",
    "missing_tags",
    "node_tags",
    "runner_tags",
    "tool_tags",
]
