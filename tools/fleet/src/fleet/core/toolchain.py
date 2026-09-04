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
    PACKAGE_MANAGERS,
    REQUIRED_PYTHON,
    REQUIRED_TOOLS,
    ToolReport,
    available_managers,
    install_command,
    missing,
    python_is_right,
)
from fleet.core import remote

#: What the toolchain probe is called under a node's stage root.
#:
#: A NAME rather than a path, resolved against the node's declared stage root
#: by its caller. ``$env:TEMP`` was tried first and is wrong:
#: :mod:`fleet.core.remote` writes through a single-quoted PowerShell literal,
#: which does not expand it, so a node would grow a directory named
#: ``$env:TEMP``. The writer creates the parent, so nothing needs to exist
#: before the very first probe.
PROBE_SCRIPT_NAME = "fleet-toolchain.ps1"

#: What the install script is called under a node's stage root.
INSTALL_SCRIPT_NAME = "fleet-install.ps1"

#: The probe, verbatim. Nothing is substituted into it by this package.
#:
#: ``--version`` is asked of each tool and the first line kept, because git and
#: poetry both print several. A tool that is present but declines to answer
#: yields an empty version rather than a failure: absence and silence are
#: different states, and only the first stops a dispatch.
PROBE_SCRIPT = """\
foreach ($tool in @('python','poetry','git','make','tar','winget','choco')) {
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
    wanted = {tool["name"] for tool in REQUIRED_TOOLS} | set(PACKAGE_MANAGERS)
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
    return parse_probe(
        remote.run_script(node["host"], f"{node['stage_root']}/{PROBE_SCRIPT_NAME}", PROBE_SCRIPT)
    )


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
        managers = available_managers(reports)
        wanted = {tool["name"]: tool for tool in REQUIRED_TOOLS}
        detail = "; ".join(
            f"{name} -- {wanted[name]['reason']} -- "
            f"{install_command(name, managers) or 'no automatic install on this node'}"
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


def install_script(names: tuple[str, ...], managers: tuple[str, ...]) -> str:
    """Render the script that installs the named tools on one node.

    Args:
        names: The tools to install, each of which must have a command for
            one of ``managers`` -- :func:`installable` is what guarantees
            that, and calling this with anything else is a caller error.
        managers: That node's available package managers, in preference
            order. Taken as an argument rather than looked up, because the
            same missing ``make`` is a winget command on lavender and a choco
            one on loki, and this function must not guess which node it is
            rendering for.

    Returns:
        The script's text, one command per tool, each preceded by an echo so
        a transcript says which command produced which failure.

    Raises:
        ValueError: If a named tool has no command for any of these
            managers. Refused rather than skipped: silently omitting it
            would report an install that covered less than it claimed, and
            the caller would then re-probe and see the tool still absent
            with no explanation.
    """
    lines: list[str] = []
    for name in names:
        command = install_command(name, managers)
        if not command:
            raise ValueError(
                f"{name!r} has no install command for managers {managers}; "
                "installable() is what filters these and it was not consulted"
            )
        lines.append(f"Write-Output 'installing {name}'")
        lines.append(command)
    return "\n".join(lines) + "\n"


def installable(reports: tuple[ToolReport, ...]) -> tuple[str, ...]:
    """Name the absent tools this node can have installed automatically.

    Node-specific in two ways at once: which tools are absent, and which
    managers are present to install them. Measured 2026-09-04, lavender had
    only winget and loki only choco, so a fleet-wide answer to this question
    does not exist.

    Args:
        reports: What a node answered, covering both the required tools and
            the package managers.

    Returns:
        The absent tools with a command for a manager this node has. A tool
        is left out when the package knows no command for it -- python and
        tar carry none, because which interpreter a machine should have is a
        decision -- and also when the node lacks the manager that command
        needs, which is a gap to report rather than a failure to raise.
    """
    managers = available_managers(reports)
    return tuple(name for name in missing(reports) if install_command(name, managers))


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
    remote.run_script(
        node["host"],
        f"{node['stage_root']}/{INSTALL_SCRIPT_NAME}",
        install_script(names, available_managers(reports)),
    )
    return names


__all__ = [
    "INSTALL_SCRIPT_NAME",
    "PROBE_SCRIPT",
    "PROBE_SCRIPT_NAME",
    "install_missing",
    "install_script",
    "installable",
    "parse_probe",
    "probe_toolchain",
    "require_ready",
]
