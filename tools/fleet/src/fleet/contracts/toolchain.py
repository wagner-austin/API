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
            gap: ``tar`` ships with the platform, and a manager that cannot
            supply the version the fleet runs (Ubuntu's Node 18.19.1) is
            left out rather than allowed to install the wrong one.

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


#: The exact Python a node is given when it has none.
#:
#: THE DECISION THIS PACKAGE ONCE REFUSED TO MAKE, made by the operator on
#: 2026-09-23 after lavender took slime checks it could not run: the
#: python.org build, at user scope, at the version every node already
#: carried (sedona, lavender and the hub all answered ``Python 3.11.9`` from
#: ``%LOCALAPPDATA%\\Programs\\Python\\Python311`` that day). User scope needs
#: no elevation and lands at that same path, ahead of the Store alias on the
#: user PATH. :data:`REQUIRED_PYTHON` is what a node is judged against; this
#: is only what an install puts there, and a later 3.11 patch still passes.
PINNED_PYTHON: Final = "3.11.9"

#: Why a python install stops on a machine where this version is already registered.
#:
#: MEASURED ON LAVENDER, 2026-09-23. GitHub Actions' setup-python had put
#: 3.11.9 in the runner's tool cache with the same python.org installer, and
#: its component packages were registered machine-wide. The per-user winget
#: install of the identical version was taken as a change to those same
#: products and MOVED their files to ``Python311``: the tool cache kept only
#: ``site-packages`` and ``Scripts``, and every Windows Python job on that
#: runner failed at ``pip install poetry`` until the files were copied back.
#: A registration without a usable python on the build PATH means an
#: interpreter exists somewhere a second install would uproot, so the command
#: stops with the reason instead, and the fix is to put that interpreter on
#: the PATH. A node with python already on its PATH never reaches this, since
#: it is not missing one.
PYTHON_REGISTERED_MESSAGE: Final = (
    f"Python {PINNED_PYTHON} is already registered on this machine (a setup-python tool "
    "cache is one such copy); installing it again moves that copy's files, as it did on "
    "lavender 2026-09-23. Put the registered interpreter on the build account's PATH instead."
)

#: The PowerShell that stops an install with :data:`PYTHON_REGISTERED_MESSAGE`.
#:
#: One plain line on stderr rather than ``Write-Error``, whose record wraps the
#: text in a category and a position, so the node's transcript and the failed
#: install's AppError carry the sentence and nothing else.
PYTHON_REGISTERED_GUARD: Final = (
    "if (Get-ItemProperty "
    "'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*',"
    "'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\*' "
    "-ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -like "
    f"'Python {PINNED_PYTHON} Core Interpreter*' }}) {{ [Console]::Error.WriteLine("
    "'" + PYTHON_REGISTERED_MESSAGE.replace("'", "''") + "'); exit 1 }; "
)

#: Everything a node needs to run a project's ``make check``.
#:
#: ``tar`` is on the list even though every probed node had it, because the
#: staging transport depends on it and a node acquired later may not. A
#: requirement that is documented only by being satisfied is not documented.
#:
#: ``node`` joined on 2026-09-23 for the same reason: every node answered
#: v24 that day, but nothing asked, and slime's check (tsc, eslint, vitest)
#: cannot start without it. Presence is the requirement, like git and make.
#: The winget and choco packages are each manager's current LTS, which was
#: 24 on both when this was written (OpenJS.NodeJS.LTS 24.19.0, nodejs-lts
#: 24.21.0).
REQUIRED_TOOLS: Final[tuple[RequiredTool, ...]] = (
    RequiredTool(
        name="python",
        reason="every project pins Python 3.11; poetry builds its venv from it",
        install={
            "winget": (
                f"{PYTHON_REGISTERED_GUARD}winget install --id Python.Python.3.11 "
                f"--version {PINNED_PYTHON} -e --scope user --source winget --silent "
                "--accept-package-agreements --accept-source-agreements --disable-interactivity"
            ),
            "choco": (
                f"{PYTHON_REGISTERED_GUARD}choco install python311 --version {PINNED_PYTHON} -y"
            ),
        },
    ),
    RequiredTool(
        name="poetry",
        reason="every Makefile's lint and test targets run poetry lock and poetry sync",
        install={
            "pip": "python -m pip install --user poetry",
            "pipx": "pipx install poetry",
        },
    ),
    RequiredTool(
        name="git",
        reason="some suites read the repository state they are testing",
        install={
            "winget": "winget install --id Git.Git -e --source winget --accept-source-agreements",
            "choco": "choco install git -y",
            "apt-get": "sudo apt-get install -y git",
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
            "apt-get": "sudo apt-get install -y make",
        },
    ),
    RequiredTool(
        name="node",
        reason="slime and the TypeScript packages run tsc, eslint and vitest under node",
        install={
            "winget": (
                "winget install --id OpenJS.NodeJS.LTS -e --source winget --silent "
                "--accept-package-agreements --accept-source-agreements --disable-interactivity"
            ),
            "choco": "choco install nodejs-lts -y",
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
#: ``winget`` before ``choco`` because it ships with Windows and needs no
#: elevation for a user-scope install, while choco is a deliberate
#: installation somebody made. Measured 2026-09-04: sedona has both, lavender
#: only winget, loki only choco -- so the order decides only sedona, and
#: either would work there.
#:
#: ``pipx`` and ``apt-get`` are the Linux pair, added with the first Linux
#: node (diphtheria, 2026-09-20). ``pipx`` is how poetry documents its own
#: install. ``apt-get`` is spelled with ``sudo`` because a package install is
#: root's act on that platform; the node's account must be allowed it, and
#: the command failing on a password prompt is the honest answer when it is
#: not. A Windows probe never reports these two and a Linux probe never
#: reports the Windows three, so the order only ever decides among managers
#: one platform has.
PACKAGE_MANAGERS: Final[tuple[str, ...]] = ("pip", "pipx", "winget", "choco", "apt-get")

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
    "PINNED_PYTHON",
    "PYTHON_REGISTERED_GUARD",
    "PYTHON_REGISTERED_MESSAGE",
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
