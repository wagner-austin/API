"""Asking a node what it has installed, and installing what it does not.

THE PROBE IS A CONSTANT SCRIPT, sent and run by path like every other remote
act here -- see :mod:`fleet.core.remote` for the two failed attempts that made
that a rule. It emits ``name=present=version`` lines, one per tool, parsed
strictly.

WHY IT ASKS FOR A VERSION AND NOT JUST PRESENCE. ``loki`` has poetry installed
under Python 3.12 while every project here pins 3.11. A presence check would
call that node ready, it would stage, and the build would fail resolving a
lockfile against the wrong interpreter -- which reads as a broken project
rather than a misconfigured node.

INSTALLING IS A SEPARATE FUNCTION AND A SEPARATE FLAG. These are other
people's machines, and a dispatcher that installed software because a build
wanted it would be doing the thing this whole package exists to stop: acting
on somebody else's computer without their knowing. :func:`install_missing`
exists, is explicit, and is never called by a dispatch.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.node import NodeConfig
from fleet.contracts.toolchain import (
    REQUIRED_PYTHON,
    REQUIRED_TOOLS,
    ToolReport,
    missing,
    python_is_right,
)
from fleet.core import remote

#: Where the toolchain probe is written on the node.
#:
#: Under the node's temp directory rather than the stage root, because this
#: runs BEFORE a node is known to be usable and must not depend on a directory
#: the bootstrap has not created yet.
PROBE_REMOTE_PATH = "$env:TEMP\\fleet-toolchain.ps1"

#: The probe, verbatim. Nothing is substituted into it by this package.
#:
#: ``--version`` is asked of each tool and the first line kept, because git and
#: poetry both print several. A tool that is present but declines to answer
#: yields an empty version rather than a failure: absence and silence are
#: different states, and only the first stops a dispatch.
PROBE_SCRIPT = """\
foreach ($tool in @('python','poetry','git','make','tar')) {
  $found = Get-Command $tool -ErrorAction SilentlyContinue
  if ($found) {
    $raw = (& $tool --version 2>&1 | Select-Object -First 1)
    $text = ($raw | Out-String).Trim() -replace '[\\r\\n]', ' '
    "$tool=yes=$text"
  } else {
    "$tool=no="
  }
}
"""


def parse_probe(output: str) -> tuple[ToolReport, ...]:
    """Read a toolchain probe's output into one report per tool.

    Args:
        output: The probe script's standard output.

    Returns:
        One report per line that parsed, in the order the node emitted them.

    Raises:
        AppError: With ``NODE_TOOL_MISSING`` when the output names none of
            the required tools. That is not a node without tools -- it is a
            probe that did not run, and reporting every tool as absent would
            send the reader to install five things that are already there.
    """
    reports: list[ToolReport] = []
    wanted = {tool["name"] for tool in REQUIRED_TOOLS}
    for line in output.splitlines():
        parts = line.strip().split("=", 2)
        if len(parts) != 3 or parts[0] not in wanted:
            continue
        reports.append(
            ToolReport(name=parts[0], present=parts[1] == "yes", version=parts[2].strip())
        )
    if not reports:
        raise AppError(
            FleetErrorCode.NODE_TOOL_MISSING,
            f"a toolchain probe returned nothing recognisable, so the node was never asked: "
            f"{output.strip()!r}",
        )
    return tuple(reports)


def probe_toolchain(node: NodeConfig) -> tuple[ToolReport, ...]:
    """Ask a node what it has installed.

    Args:
        node: The node to probe.

    Returns:
        One report per required tool.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if ssh cannot reach it,
            ``DISPATCH_FAILED`` if the probe exits non-zero, or
            ``NODE_TOOL_MISSING`` if its answer cannot be read.
    """
    return parse_probe(remote.run_script(node["host"], PROBE_REMOTE_PATH, PROBE_SCRIPT))


def require_ready(node_name: str, node: NodeConfig, reports: tuple[ToolReport, ...]) -> None:
    """Refuse a node that cannot run a build.

    Args:
        node_name: The node's workspace name.
        node: Its declaration, for the host in the message.
        reports: What it answered.

    Raises:
        AppError: With ``NODE_TOOL_MISSING`` when a required tool is absent,
            naming every one and what would install it, or
            ``NODE_PYTHON_MISMATCH`` when everything is present but the
            interpreter is the wrong minor version. Two codes because the
            fixes differ: one is a package manager, the other is a decision
            about which Python that machine should carry.
    """
    absent = missing(reports)
    if absent:
        wanted = {tool["name"]: tool for tool in REQUIRED_TOOLS}
        detail = "; ".join(
            f"{name} -- {wanted[name]['reason']} -- "
            f"{wanted[name]['install'] or 'no automatic install'}"
            for name in absent
            if name in wanted
        )
        raise AppError(
            FleetErrorCode.NODE_TOOL_MISSING,
            f"{node_name} ({node['host']}) cannot run a build: {detail}",
        )
    if not python_is_right(reports):
        raise AppError(
            FleetErrorCode.NODE_PYTHON_MISMATCH,
            f"{node_name} ({node['host']}) reports Python "
            f"{_python_version(reports)!r} where {REQUIRED_PYTHON} is required; every project "
            "resolves its lockfile against that minor version, so a build here would fail "
            "resolving rather than testing",
        )


def _python_version(reports: tuple[ToolReport, ...]) -> str:
    """Read the node's reported Python version.

    Args:
        reports: What it answered.

    Returns:
        The version string, or ``unknown`` when it did not say.
    """
    for report in reports:
        if report["name"] == "python":
            return report["version"] or "unknown"
    return "unknown"


def install_script(names: tuple[str, ...]) -> str:
    """Render the script that installs the named tools on a node.

    Args:
        names: The tools to install, which must all carry an install command.

    Returns:
        The script's text, one command per tool, each echoing what it is
        about to do so the transcript says which command produced which
        failure.
    """
    wanted = {tool["name"]: tool for tool in REQUIRED_TOOLS}
    lines = []
    for name in names:
        lines.append(f"Write-Output 'installing {name}'")
        lines.append(wanted[name]["install"])
    return "\n".join(lines) + "\n"


def installable(reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Name the absent tools this package knows how to install.

    Args:
        reports: What a node answered.

    Returns:
        The absent tools carrying an install command. A tool with no command
        is left out rather than reported as a failure: Python and tar are
        decisions about what a machine should carry, not packages to add
        because a build wanted them.
    """
    wanted = {tool["name"]: tool for tool in REQUIRED_TOOLS}
    return tuple(name for name in missing(reports) if name in wanted and wanted[name]["install"])


def install_missing(node: NodeConfig, reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Install what a node is missing and this package can supply.

    Args:
        node: The node to install on.
        reports: What it answered when probed.

    Returns:
        The tools that were installed. Empty when there was nothing to do,
        which is a real answer rather than an error -- a node may be missing
        only things that have no automatic install.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` when the
            install command itself fails, carrying the node's own stderr. Not
            softened: a half-installed node is worse than an untouched one
            because it looks ready.
    """
    names = installable(reports)
    if not names:
        return ()
    remote.run_script(node["host"], "$env:TEMP\\fleet-install.ps1", install_script(names))
    return names


__all__ = [
    "PROBE_REMOTE_PATH",
    "PROBE_SCRIPT",
    "install_missing",
    "install_script",
    "installable",
    "parse_probe",
    "probe_toolchain",
    "require_ready",
]
