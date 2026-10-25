"""What a node must already have before work can be dispatched to it.

THE STATE THIS FILE WAS WRITTEN AGAINST, measured 2026-09-04 before any of it
was fixed:

    node      python3.11  poetry  git   make   winget  choco
    sedona    Store       2.2.1   2.43  NO     yes     yes
    lavender  3.11.9      NO      NO    NO     yes     NO
    loki      3.11.9      2.1.3   2.50  yes    NO      yes

ONE NODE OF THREE could have run a ``make check``. A dispatcher that assumed
otherwise would stage a whole monorepo, launch, and fail on the recipe's first
line -- having spent the transfer to learn what one probe answers instantly.

The fleet was standardised the same day and now reads 3.11.9 / 2.4.2 / 2.55.0
/ 4.4.1 across all three. THAT IS EXACTLY WHY THIS FILE STAYS. A fleet drifts
the moment somebody installs something, and the table above is what drift
looked like the first time anyone checked. The probe is the thing that notices
the second time.

WHY THE PYTHON VERSION IS PART OF THE REQUIREMENT AND NOT JUST THE BINARY.
``loki`` had poetry installed under Python 3.12 while every project here pins
``^3.11`` -- WHICH ACCEPTS 3.12. So poetry would have built a 3.12 venv and
resolved every lockfile against the wrong minor version, silently, and the
build would have failed resolving rather than testing. Presence says nothing;
the version says everything.

AND WHY THE INSTALL KIND MATTERS TOO. ``sedona``'s Python was the Microsoft
Store build, which sandboxes ``%LOCALAPPDATA%`` writes and has broken poetry
venv creation before. It was replaced with a python.org install at the same
path the other nodes use. A probe that only asked "is there a python" would
have called that node ready.

WHY THIS REFUSES RATHER THAN INSTALLS BY DEFAULT. These are other people's
machines. Installing software on one is not something a dispatcher should do
because a build wanted it, so :mod:`fleet.cli.bootstrap` reports and names the
command, and installs only when asked in so many words.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_str,
)
from typing_extensions import TypedDict

from fleet.contracts.tagged_tools import TAGGED_TOOLS, RequiredTool


class ToolReport(TypedDict):
    """What one node answered about one tool.

    Attributes:
        name: The executable, or the probe line's name.
        present: Whether it is on the node's PATH.
        version: What it reported, or an empty string when absent or when it
            declines to say. Recorded because presence is not the whole
            question -- ``loki``'s poetry is present and runs on the wrong
            Python.
    """

    name: str
    present: bool
    version: str


#: The Poetry every install puts on a machine, and every host that commits or
#: bakes a working tree (the hub, the session host, diphtheria) runs: the one
#: every committed lock was written by. Measured 2026-10-17 (MCPs board task
#: ce698bfb): 2.4.2 re-locks tools/fleet byte for byte, 2.5.1 rewrites its
#: header and spells pydantic's exclusions as ``!=``, and a path-scoped commit
#: then carries that rewrite. Readers outside this package take it from
#: ``rendered/toolchain-pins.json`` (:mod:`fleet.core.toolchain_document`).
PINNED_POETRY: Final = "2.4.2"

#: Everything a node needs to run a project's ``make check``.
#:
#: ``tar`` is on the list even though every probed node had it, because the
#: staging transport depends on it and a node acquired later may not. A
#: requirement that is documented only by being satisfied is not documented.
#:
#: ``node`` joined on 2026-09-23 for the same reason: every node answered
#: v24 that day, but nothing asked, and slime's check (tsc, eslint, vitest)
#: cannot start without it. Presence is the requirement, like git and make.
#:
#: ``python`` and ``node`` carry no install command: winget and choco were
#: the only managers that installed them, and those left with the fleet's
#: last Windows node (board task e705f08c); a Linux node gets both from its
#: provisioning, never from a dispatch.
#:
#: ``ffmpeg`` is NOT on this list, although it was from 2026-10-01 (MCPs board
#: task 512e7bf8) to 2026-10-02: only grandma-api's check uses it, and as a
#: tool every build required it idled a whole node for one project, which is
#: how pendragon claimed nothing on every tick of 2026-10-02 until 02:48Z
#: while tag-free jobs waited (MCPs board task 939ec5c7). It is a tag now:
#: :data:`fleet.contracts.tagged_tools.TAGGED_TOOLS`.
REQUIRED_TOOLS: Final[tuple[RequiredTool, ...]] = (
    RequiredTool(
        name="python",
        reason="every project pins Python 3.11; poetry builds its venv from it",
        install={},
        uninstall={},
    ),
    RequiredTool(
        name="poetry",
        reason="every Makefile's lint and test targets run poetry lock and poetry sync",
        install={
            "pip": f"python -m pip install --user poetry=={PINNED_POETRY}",
            "pipx": f"pipx install poetry=={PINNED_POETRY}",
        },
        uninstall={
            "pip": "python -m pip uninstall -y poetry",
            "pipx": "pipx uninstall poetry",
        },
    ),
    RequiredTool(
        name="git",
        reason="some suites read the repository state they are testing",
        install={"apt-get": "sudo apt-get install -y git"},
        uninstall={"apt-get": "sudo apt-get remove -y git"},
    ),
    RequiredTool(
        name="make",
        reason="make check is the entry point for every project in this monorepo",
        install={"apt-get": "sudo apt-get install -y make"},
        uninstall={"apt-get": "sudo apt-get remove -y make"},
    ),
    RequiredTool(
        name="node",
        reason="slime and the TypeScript packages run tsc, eslint and vitest under node",
        install={},
        uninstall={},
    ),
    RequiredTool(
        name="tar",
        reason="staging sends a gzipped tar and the node unpacks it",
        install={},
        uninstall={},
    ),
)

#: The package managers a node is asked about, in the order they are preferred.
#:
#: ``pip`` first because poetry installs through the interpreter's own pip
#: and needs no system package manager at all -- and a node that has no
#: Python cannot run a build regardless, so nothing is lost by preferring it.
#: It was keyed ``python`` until 2026-09-20 and reported by the interpreter's
#: own line; it is its own line now (``python -m pip --version`` answering)
#: because a Linux node reports its interpreter under ``python`` too, and
#: there the user-site install this manager runs is refused outright:
#: Ubuntu 24.04's interpreter is externally managed (PEP 668). Splitting the
#: manager from the tool is what lets that node report the interpreter it has
#: without being offered an install command that cannot work on it.
#:
#: ``pipx`` and ``apt-get`` were added with the first Linux node
#: (diphtheria, 2026-09-20). ``pipx`` is how poetry documents its own
#: install. ``apt-get`` is spelled with ``sudo`` because a package install is
#: root's act; the node's account must be allowed it, and the command failing
#: on a password prompt is the honest answer when it is not. ``winget`` and
#: ``choco``, the Windows pair, left with the fleet's last Windows node
#: (board task e705f08c).
PACKAGE_MANAGERS: Final[tuple[str, ...]] = ("pip", "pipx", "apt-get")

#: The Python a project's environment is built from.
#:
#: A prefix rather than an exact string, so a patch release does not fail a
#: node. The minor version is the part that matters: 3.12 resolves a different
#: dependency set from the lockfile every project here pins.
REQUIRED_PYTHON = "3.11"

#: The oldest Node major a project's check is built against.
#:
#: A floor rather than a prefix like :data:`REQUIRED_PYTHON`, because the
#: requirement is one: MCPs declares ``engines.node >=24`` and its CI runs 24,
#: and a newer major only ever meets it. ``node`` was judged on presence
#: alone until 2026-09-25, when serendipity answered ``v18.13.0`` (npm
#: 8.19.3), was called ready, claimed MCPs/packages/wiki-search and failed
#: rebuilding hnswlib-node under node-gyp before any test ran (dispatch job
#: de32d61e, MCPs board task fd5cabfa).
REQUIRED_NODE_MAJOR = 24


def missing(reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Name the REQUIRED tools a node does not have.

    Filtered to :data:`REQUIRED_TOOLS` rather than returning everything the
    probe found absent, because the probe also asks about package managers
    and a node is not required to have any particular one. Unfiltered, a
    node without pipx would be reported as missing a tool and refused,
    though it can build perfectly well.

    Args:
        reports: What the node answered, which covers the required tools and
            the package managers together.

    Returns:
        The absent required tools' names, in the order they were reported.
    """
    required = {tool["name"] for tool in REQUIRED_TOOLS}
    return tuple(
        report["name"] for report in reports if report["name"] in required and not report["present"]
    )


def version_number(reported: str) -> str:
    """Pull the version out of what a tool printed when asked.

    THE FIRST VERSION OF THIS DID NOT EXIST and its absence was a real bug:
    ``python --version`` prints ``Python 3.11.9``, so comparing the whole
    string against ``3.11`` reported every node as carrying the wrong
    interpreter -- including the three that carry the right one. Its own test
    caught it.

    The LAST whitespace-separated token, because every tool here leads with
    its own name and some add more: ``git version 2.43.0``,
    ``Poetry (version 2.1.3)``. A tool that printed only a number is
    unaffected, which is what makes this safe to apply to all of them.

    Args:
        reported: What the tool printed, already stripped.

    Returns:
        The trailing token, with any wrapping parenthesis removed, or an
        empty string when nothing was reported.
    """
    if not reported:
        return ""
    return reported.split()[-1].strip("()")


def python_is_right(reports: tuple[ToolReport, ...]) -> bool:
    """Whether the node's Python is the one projects are built against.

    Args:
        reports: What the node answered.

    Returns:
        True when a ``python`` report is present and its version number
        begins with :data:`REQUIRED_PYTHON`. A node that reported no Python
        at all is False here as well as in :func:`missing`, which is
        deliberate: the two questions have the same answer and the caller
        should not have to ask both to learn the node is unusable.
    """
    for report in reports:
        if report["name"] == "python":
            return report["present"] and version_number(report["version"]).startswith(
                REQUIRED_PYTHON
            )
    return False


def node_is_right(reports: tuple[ToolReport, ...]) -> bool:
    """Whether the node's Node.js is at least :data:`REQUIRED_NODE_MAJOR`.

    Args:
        reports: What the node answered.

    Returns:
        True when a ``node`` report is present and its version (``v24.20.0``,
        the leading ``v`` optional) has a whole-number major at or above the
        floor. False for a node that reported no Node, as
        :func:`python_is_right` is for Python, and for a version whose major
        is not a number, which no real ``node --version`` prints.
    """
    for report in reports:
        if report["name"] == "node":
            major = version_number(report["version"]).removeprefix("v").split(".")[0]
            return report["present"] and major.isdigit() and int(major) >= REQUIRED_NODE_MAJOR
    return False


def is_ready(reports: tuple[ToolReport, ...]) -> bool:
    """Whether a node can run a build as it stands.

    Args:
        reports: What it answered when probed.

    Returns:
        True when nothing is absent, the interpreter is the right minor
        version and Node.js is at the floor. All three, because a node with
        every tool present and the wrong runtime fails at lockfile resolution
        or at a native build, which reads as a broken project rather than a
        misconfigured node.
    """
    return not missing(reports) and python_is_right(reports) and node_is_right(reports)


def available_managers(reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Name the package managers this node actually has, in preference order.

    Args:
        reports: What the node answered, which includes the managers it was
            asked about as well as the tools it must have.

    Returns:
        The present managers, ordered by :data:`PACKAGE_MANAGERS`. Empty when
        the node has none, which is not an error -- it means nothing can be
        installed there automatically and the gap has to be closed by hand.
    """
    present = {report["name"] for report in reports if report["present"]}
    return tuple(manager for manager in PACKAGE_MANAGERS if manager in present)


def install_command(tool: str, managers: tuple[str, ...]) -> str:
    """Choose how to install one tool on a node with these managers.

    Args:
        tool: The tool's name.
        managers: The node's available managers, in preference order.

    Returns:
        The first command whose manager the node has, or an empty string when
        the tool has no command for any of them. Empty covers both cases a
        caller must not conflate with failure: a tool this package never
        installs, and a node whose managers do not cover it.
    """
    chosen = _chosen_manager(tool, managers)
    return "" if chosen is None else chosen[0]["install"][chosen[1]]


def uninstall_command(tool: str, managers: tuple[str, ...]) -> str:
    """Choose how to remove one tool that :func:`install_command` installed.

    Args:
        tool: The tool's name.
        managers: The node's available managers, in preference order.

    Returns:
        The removal command for the SAME manager :func:`install_command`
        chooses on these managers, so a rollback undoes the install that
        ran rather than asking another manager about a package it never
        placed; empty exactly when :func:`install_command` is.
    """
    chosen = _chosen_manager(tool, managers)
    return "" if chosen is None else chosen[0]["uninstall"][chosen[1]]


def _chosen_manager(tool: str, managers: tuple[str, ...]) -> tuple[RequiredTool, str] | None:
    """Find a tool's row and the first of these managers that installs it.

    Args:
        tool: The tool's name.
        managers: The node's available managers, in preference order.

    Returns:
        The tool's row and the chosen manager, or None when the tool has no
        install command for any of them.
    """
    for required in REQUIRED_TOOLS + TAGGED_TOOLS:
        if required["name"] != tool:
            continue
        for manager in managers:
            if required["install"].get(manager):
                return required, manager
    return None


def describe_gap(node: str, reports: tuple[ToolReport, ...]) -> str:
    """Render what stands between a node and its first dispatch.

    Args:
        node: The node's workspace name.
        reports: What it answered.

    Returns:
        One line naming what is absent and what would install it ON THIS
        NODE, or a line saying the node is ready. Node-specific because a
        node has the package managers it has, and one without ``apt-get``
        gets no command for ``make``.
    """
    if is_ready(reports):
        return f"{node}: ready"
    absent = missing(reports)
    managers = available_managers(reports)
    wanted = {tool["name"] for tool in REQUIRED_TOOLS}
    parts = [
        f"{name} ({install_command(name, managers) or 'install by hand'})"
        for name in absent
        if name in wanted
    ]
    if not python_is_right(reports) and "python" not in absent:
        parts.append(f"python {REQUIRED_PYTHON} (found {reported_version(reports, 'python')})")
    if not node_is_right(reports) and "node" not in absent:
        parts.append(f"node {REQUIRED_NODE_MAJOR}+ (found {reported_version(reports, 'node')})")
    return f"{node}: missing {', '.join(parts)}"


def reported_version(reports: tuple[ToolReport, ...], name: str) -> str:
    """Read one tool's reported version.

    Args:
        reports: What the node answered.
        name: The tool to look up.

    Returns:
        Its version, or ``unknown`` when it did not say. A word rather than
        an empty string, because this lands in the middle of a sentence.
    """
    for report in reports:
        if report["name"] == name:
            return report["version"] or "unknown"
    return "unknown"


def encode_tool_report(report: ToolReport) -> JSONObject:
    """Encode one tool's report.

    Args:
        report: The report to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "name": report["name"],
        "present": report["present"],
        "version": report["version"],
    }


def decode_tool_report(value: JSONValue) -> ToolReport:
    """Decode and validate one tool's report.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated report.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or a tool is marked absent while carrying a version. A
            version is something only a present tool can have reported, so
            the combination means the two fields came from different reads
            and neither can be trusted.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"tool report must be a JSON object, got {type(value).__name__}")
    present = require_bool(value, "present")
    version = require_str(value, "version")
    if not present and version:
        raise JSONTypeError(
            f"tool report says absent but carries version {version!r}; only a present tool "
            "can have reported one, so the two fields came from different reads"
        )
    return ToolReport(name=require_str(value, "name"), present=present, version=version)


__all__ = [
    "PACKAGE_MANAGERS",
    "PINNED_POETRY",
    "REQUIRED_NODE_MAJOR",
    "REQUIRED_PYTHON",
    "REQUIRED_TOOLS",
    "ToolReport",
    "available_managers",
    "decode_tool_report",
    "describe_gap",
    "encode_tool_report",
    "install_command",
    "is_ready",
    "missing",
    "node_is_right",
    "python_is_right",
    "reported_version",
    "uninstall_command",
    "version_number",
]
