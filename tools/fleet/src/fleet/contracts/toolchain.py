"""What a node must already have before work can be dispatched to it.

MEASURED 2026-09-04, and the reason this file exists rather than an assumption
that a machine on the tailnet is ready:

    node      python3.11  poetry  git   make   tar
    sedona    3.11.9      2.2.1   2.43  NO     yes
    lavender  3.11.9      NO      NO    NO     yes
    loki      3.11.9      2.1.3   2.50  yes    yes

One node of three could have run a ``make check`` at all. A dispatcher that
assumed otherwise would stage a whole monorepo, launch, and fail on the first
line of the recipe -- having spent the transfer to learn something one probe
answers instantly.

WHY THE PYTHON VERSION IS PART OF THE REQUIREMENT AND NOT JUST THE BINARY.
``loki`` has poetry installed under Python 3.12 while every project here pins
3.11. Poetry itself runs fine on 3.12 and manages a 3.11 environment, so the
presence of a ``python`` says nothing useful on its own -- the version does.

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
    """One thing a node must have, and how to get it.

    Attributes:
        name: The executable, as it is spelled on a PATH.
        reason: Why a dispatch needs it. Carried so a refusal explains
            itself rather than naming a binary and leaving the reader to
            infer what it was for.
        install: The command that installs it on a Windows node, or an empty
            string for something that cannot be installed by this package.
            Empty is a real state: a node without a working Python 3.11 needs
            a decision, not a package manager invocation.
    """

    name: str
    reason: str
    install: str


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
        install="",
    ),
    RequiredTool(
        name="poetry",
        reason="every Makefile's lint and test targets run poetry lock and poetry sync",
        install="python -m pip install --user poetry",
    ),
    RequiredTool(
        name="git",
        reason="some suites read the repository state they are testing",
        install="choco install git -y",
    ),
    RequiredTool(
        name="make",
        reason="make check is the entry point for every project in this monorepo",
        install="choco install make -y",
    ),
    RequiredTool(
        name="tar",
        reason="staging sends a gzipped tar and the node unpacks it",
        install="",
    ),
)

#: The Python a project's environment is built from.
#:
#: A prefix rather than an exact string, so a patch release does not fail a
#: node. The minor version is the part that matters: 3.12 resolves a different
#: dependency set from the lockfile every project here pins.
REQUIRED_PYTHON = "3.11"


def missing(reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Name the tools a node does not have.

    Args:
        reports: What the node answered, one entry per required tool.

    Returns:
        The absent tools' names, in the order they were reported.
    """
    return tuple(report["name"] for report in reports if not report["present"])


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


def describe_gap(node: str, reports: tuple[ToolReport, ...]) -> str:
    """Render what stands between a node and its first dispatch.

    Args:
        node: The node's workspace name.
        reports: What it answered.

    Returns:
        One line naming what is absent and what would install it, or a line
        saying the node is ready. The install commands are included because
        the reader's next question is always how to fix it, and sending them
        to a README for that is how a gap stays open.
    """
    absent = missing(reports)
    if not absent and python_is_right(reports):
        return f"{node}: ready"
    wanted = {tool["name"]: tool for tool in REQUIRED_TOOLS}
    parts = [
        f"{name} ({wanted[name]['install'] or 'install by hand'})"
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
    "REQUIRED_PYTHON",
    "REQUIRED_TOOLS",
    "RequiredTool",
    "ToolReport",
    "decode_tool_report",
    "describe_gap",
    "encode_tool_report",
    "missing",
    "python_is_right",
    "version_number",
]
