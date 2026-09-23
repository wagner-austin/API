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

from fleet.contracts.node import NodeConfig, NodePlatform
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
from fleet.core import dialect, names, remote


def read_reports(output: str) -> tuple[ToolReport, ...]:
    """Read a toolchain probe's output into one report per recognised line.

    Args:
        output: The probe script's standard output.

    Returns:
        One report per line naming a required tool or a package manager, in
        the order the node emitted them. Empty when no line did.
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
    return tuple(reports)


def _unrecognised(output: str) -> str:
    """Explain an answer that names no tool.

    Args:
        output: The probe script's standard output.

    Returns:
        The explanation, quoting the whole answer.
    """
    return (
        f"a toolchain probe returned nothing recognisable, so the node was never asked: "
        f"{output.strip()!r}"
    )


def parse_probe(output: str) -> tuple[ToolReport, ...]:
    """Read a toolchain probe's output into one report per tool.

    The raising boundary over :func:`read_reports`.

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
    reports = read_reports(output)
    if not reports:
        raise AppError(FleetErrorCode.NODE_TOOL_MISSING, _unrecognised(output))
    return reports


def attempt_toolchain(node: NodeConfig) -> tuple[ToolReport, ...] | remote.RemoteFailure:
    """Ask a node what it has installed, reporting failure as a value.

    THE VALUE FORM EXISTS FOR THE NODE RUNNER. A runner deciding whether to
    claim this tick treats a node that did not answer the way it treats one
    that answered "python absent": it claims nothing and says why, and a
    tick that raised on either would stop on the condition it exists to
    report. :func:`probe_toolchain` raises on top of this for a caller that
    named the node, ``fleet-bootstrap``.

    Args:
        node: The node to probe.

    Returns:
        Its reports, never empty; or the typed reason there are none: the
        transport's ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED``, or
        ``NODE_TOOL_MISSING`` when it answered with no line this package
        recognises.
    """
    # Under the stage root rather than the node's TEMP: ``$env:TEMP`` was
    # tried first and is wrong, because the writer's single-quoted literal
    # does not expand it and a node would grow a directory of that name. The
    # writer creates the parent, so nothing needs to exist before the very
    # first probe.
    spoken = dialect.for_platform(node["platform"])
    outcome = remote.attempt_script(
        node["host"],
        spoken.script_path(node["stage_root"], names.TOOLCHAIN_PROBE_STEM),
        spoken.toolchain_probe_script(),
        platform=node["platform"],
    )
    failure = outcome["failure"]
    if failure is not None:
        return failure
    reports = read_reports(outcome["output"])
    if not reports:
        return remote.RemoteFailure(
            code=FleetErrorCode.NODE_TOOL_MISSING,
            message=f"{node['host']}: {_unrecognised(outcome['output'])}",
        )
    return reports


def probe_toolchain(node: NodeConfig) -> tuple[ToolReport, ...]:
    """Ask a node what it has installed.

    The raising boundary over :func:`attempt_toolchain`.

    Args:
        node: The node to probe.

    Returns:
        One report per required tool.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` if ssh cannot reach it,
            ``DISPATCH_FAILED`` if the probe exits non-zero, or
            ``NODE_TOOL_MISSING`` if its answer cannot be read.
    """
    outcome = attempt_toolchain(node)
    if isinstance(outcome, tuple):
        return outcome
    raise AppError(outcome["code"], outcome["message"])


def readiness_gap(
    node_name: str, node: NodeConfig, reports: tuple[ToolReport, ...]
) -> AppError[FleetErrorCode] | None:
    """What stands between a node and a build, as a value.

    Args:
        node_name: The node's workspace name.
        node: Its declaration, for the host in the message.
        reports: What it answered.

    Returns:
        ``None`` when the node can build. Otherwise the refusal, not
        raised: ``NODE_TOOL_MISSING`` naming every absent tool, why a build
        needs it and what would install it on THIS node, or
        ``NODE_PYTHON_MISMATCH`` when everything is present but the
        interpreter is the wrong minor version. Two codes because the fixes
        differ: one is a package manager, the other is a decision about which
        Python that machine should carry.
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
        return AppError(
            FleetErrorCode.NODE_TOOL_MISSING,
            f"{node_name} ({node['host']}) cannot run a build: {detail}",
        )
    if not python_is_right(reports):
        return AppError(
            FleetErrorCode.NODE_PYTHON_MISMATCH,
            f"{node_name} ({node['host']}) reports Python "
            f"{_python_version(reports)!r} where {REQUIRED_PYTHON} is required; every project "
            "resolves its lockfile against that minor version, so a build here would fail "
            "resolving rather than testing",
        )
    return None


def require_ready(node_name: str, node: NodeConfig, reports: tuple[ToolReport, ...]) -> None:
    """Refuse a node that cannot run a build.

    The raising boundary over :func:`readiness_gap`.

    Args:
        node_name: The node's workspace name.
        node: Its declaration, for the host in the message.
        reports: What it answered.

    Raises:
        AppError: The refusal :func:`readiness_gap` names, with its code.
    """
    gap = readiness_gap(node_name, node, reports)
    if gap is not None:
        raise gap


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


def install_script(
    tools: tuple[str, ...], managers: tuple[str, ...], *, platform: NodePlatform
) -> str:
    """Render the script that installs the named tools on one node.

    Args:
        tools: The tools to install, each of which must have a command for
            one of ``managers`` -- :func:`installable` is what guarantees
            that, and calling this with anything else is a caller error.
        managers: That node's available package managers, in preference
            order. Taken as an argument rather than looked up, because the
            same missing ``make`` is a winget command on lavender and a choco
            one on loki, and this function must not guess which node it is
            rendering for.
        platform: The node's declared platform, for the echo that precedes
            each command.

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
    spoken = dialect.for_platform(platform)
    lines: list[str] = []
    for name in tools:
        command = install_command(name, managers)
        if not command:
            raise ValueError(
                f"{name!r} has no install command for managers {managers}; "
                "installable() is what filters these and it was not consulted"
            )
        lines.append(spoken.echo_command(f"installing {name}"))
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
        is left out when the package knows no command for it on this node's
        managers -- tar carries none, and python and node none for apt-get
        -- and also when the node lacks the manager that command needs,
        which is a gap to report rather than a failure to raise.
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
    tools = installable(reports)
    if not tools:
        return ()
    spoken = dialect.for_platform(node["platform"])
    remote.run_script(
        node["host"],
        spoken.script_path(node["stage_root"], names.INSTALL_STEM),
        install_script(tools, available_managers(reports), platform=node["platform"]),
        platform=node["platform"],
    )
    return tools


__all__ = [
    "attempt_toolchain",
    "install_missing",
    "install_script",
    "installable",
    "parse_probe",
    "probe_toolchain",
    "read_reports",
    "readiness_gap",
    "require_ready",
]
