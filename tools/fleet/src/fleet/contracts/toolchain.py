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


class RequiredTool(TypedDict):
    """One thing a node must have, and how to get it on each package manager.

    Attributes:
        name: The executable, as it is spelled on a PATH.
        reason: Why a dispatch needs it. Carried so a refusal explains
            itself rather than naming a binary and leaving the reader to
            infer what it was for.
        install: The command per package manager, keyed by the manager's own
            executable name. An EMPTY MAPPING means this package does not
            install that tool at all, which is a real state rather than a
            gap: a node without a working Python 3.11 needs a decision, not a
            package manager invocation.

            A MAPPING RATHER THAN ONE COMMAND, and this was a defect before
            it was a design. The first version hardcoded ``choco install`` --
            inferred from loki's ``make`` living under
            ``C:\\ProgramData\\chocolatey``, one node generalised to three.
            Measured 2026-09-04: sedona has both managers, lavender has ONLY
            winget, loki has ONLY choco. No single command works fleet-wide,
            so the manager is chosen per node from what that node reported.
    """

    name: str
    reason: str
    install: dict[str, str]


class ToolReport(TypedDict):
    """What one node answered about one tool.

    Attributes:
        name: The executable.
        present: Whether it is on the node's PATH.
        version: What it reported, or an empty string when absent or when it
            declines to say. Recorded because presence is not the whole
            question -- ``loki``'s poetry is present and runs on the wrong
            Python.
    """

    name: str
    present: bool
    version: str


#: Everything a node needs to run a project's ``make check``.
#:
#: ``tar`` is on the list even though every probed node had it, because the
#: staging transport depends on it and a node acquired later may not. A
#: requirement that is documented only by being satisfied is not documented.
REQUIRED_TOOLS: Final[tuple[RequiredTool, ...]] = (
    RequiredTool(
        name="python",
        reason="every project pins Python 3.11; poetry builds its venv from it",
        install={},
    ),
    RequiredTool(
        name="poetry",
        reason="every Makefile's lint and test targets run poetry lock and poetry sync",
        install={"python": "python -m pip install --user poetry"},
    ),
    RequiredTool(
        name="git",
        reason="some suites read the repository state they are testing",
        install={
            "winget": "winget install --id Git.Git -e --source winget --accept-source-agreements",
            "choco": "choco install git -y",
        },
    ),
    RequiredTool(
        name="make",
        reason="make check is the entry point for every project in this monorepo",
        install={
            "winget": (
                "winget install --id GnuWin32.Make -e --source winget --accept-source-agreements"
            ),
            "choco": "choco install make -y",
        },
    ),
    RequiredTool(
        name="tar",
        reason="staging sends a gzipped tar and the node unpacks it",
        install={},
    ),
)

#: The package managers a node is asked about, in the order they are preferred.
#:
#: ``python`` first because poetry installs through its own pip and needs no
#: system package manager at all -- and a node that has no Python cannot run a
#: build regardless, so nothing is lost by preferring it.
#:
#: ``winget`` before ``choco`` because it ships with Windows and needs no
#: elevation for a user-scope install, while choco is a deliberate
#: installation somebody made. Measured 2026-09-04: sedona has both, lavender
#: only winget, loki only choco -- so the order decides only sedona, and
#: either would work there.
PACKAGE_MANAGERS: Final[tuple[str, ...]] = ("python", "winget", "choco")

#: The Python a project's environment is built from.
#:
#: A prefix rather than an exact string, so a patch release does not fail a
#: node. The minor version is the part that matters: 3.12 resolves a different
#: dependency set from the lockfile every project here pins.
REQUIRED_PYTHON = "3.11"


def missing(reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Name the REQUIRED tools a node does not have.

    Filtered to :data:`REQUIRED_TOOLS` rather than returning everything the
    probe found absent, because the probe also asks about package managers
    and a node is not required to have any particular one. Unfiltered, loki
    -- which has choco and no winget -- would be reported as missing a tool
    and refused, though it can build perfectly well.

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
    its own name and some add more: ``git version 2.50.1.windows.1``,
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
    for required in REQUIRED_TOOLS:
        if required["name"] != tool:
            continue
        for manager in managers:
            command = required["install"].get(manager)
            if command:
                return command
    return ""


def describe_gap(node: str, reports: tuple[ToolReport, ...]) -> str:
    """Render what stands between a node and its first dispatch.

    Args:
        node: The node's workspace name.
        reports: What it answered.

    Returns:
        One line naming what is absent and what would install it ON THIS
        NODE, or a line saying the node is ready. Node-specific because the
        fleet does not share a package manager: the same missing ``make`` is
        a winget command on lavender and a choco one on loki.
    """
    absent = missing(reports)
    if not absent and python_is_right(reports):
        return f"{node}: ready"
    managers = available_managers(reports)
    wanted = {tool["name"] for tool in REQUIRED_TOOLS}
    parts = [
        f"{name} ({install_command(name, managers) or 'install by hand'})"
        for name in absent
        if name in wanted
    ]
    if not python_is_right(reports) and "python" not in absent:
        parts.append(f"python {REQUIRED_PYTHON} (found {_version_of(reports, 'python')})")
    return f"{node}: missing {', '.join(parts)}"


def _version_of(reports: tuple[ToolReport, ...], name: str) -> str:
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
    "REQUIRED_PYTHON",
    "REQUIRED_TOOLS",
    "RequiredTool",
    "ToolReport",
    "available_managers",
    "decode_tool_report",
    "describe_gap",
    "encode_tool_report",
    "install_command",
    "missing",
    "python_is_right",
    "version_number",
]
