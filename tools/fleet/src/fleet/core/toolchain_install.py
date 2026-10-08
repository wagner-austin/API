"""Installing what a node lacks, verifying it landed, and rolling back what did not.

Split from :mod:`fleet.core.toolchain` when the install gained its verify and
its rollback (board task cc7222ca), which would have taken that module past
the 600-line ceiling. That module asks a node what it has; this one changes
it.

INSTALLING IS A SEPARATE FUNCTION AND A SEPARATE FLAG. These are other
people's machines, and a dispatcher that installed software because a build
wanted it would be doing the thing this whole package exists to stop: acting
on somebody else's computer without their knowing. :func:`install_missing`
exists, is explicit, and is never called by a dispatch. When it runs and
does not finish, it removes what it landed, so the machine is left as its
owner last saw it.
"""

from __future__ import annotations

from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.contracts.toolchain import (
    ToolReport,
    available_managers,
    install_command,
    missing,
    uninstall_command,
)
from fleet.core import dialect, names, remote
from fleet.core.toolchain import probe_toolchain


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
    commands = tuple((name, install_command(name, managers)) for name in tools)
    return _manager_script(commands, managers, verb="install", platform=platform)


def uninstall_script(
    tools: tuple[str, ...], managers: tuple[str, ...], *, platform: NodePlatform
) -> str:
    """Render the script that removes tools :func:`install_script` installed.

    Args:
        tools: The tools to remove, each installable on ``managers``.
        managers: The node's available package managers, in preference
            order: the same tuple the install was rendered with, so each
            removal goes through the manager that installed the tool.
        platform: The node's declared platform.

    Returns:
        The script's text, one echo and one command per tool.

    Raises:
        ValueError: If a named tool has no command for any of these managers,
            as :func:`install_script` refuses it.
    """
    commands = tuple((name, uninstall_command(name, managers)) for name in tools)
    return _manager_script(commands, managers, verb="uninstall", platform=platform)


def _manager_script(
    commands: tuple[tuple[str, str], ...],
    managers: tuple[str, ...],
    *,
    verb: str,
    platform: NodePlatform,
) -> str:
    """Render one package-manager command per tool, each after an echo.

    Args:
        commands: Each tool's name and its command, empty when it has none.
        managers: The managers the commands were chosen from, for the refusal.
        verb: ``install`` or ``uninstall``, for the echo and the refusal.
        platform: The node's declared platform.

    Returns:
        The script's text.

    Raises:
        ValueError: If a tool's command is empty.
    """
    spoken = dialect.for_platform(platform)
    lines: list[str] = []
    for name, command in commands:
        if not command:
            raise ValueError(
                f"{name!r} has no {verb} command for managers {managers}; "
                "installable() is what filters these and it was not consulted"
            )
        lines.append(spoken.echo_command(f"{verb}ing {name}"))
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


def install_missing(
    node: NodeConfig, reports: tuple[ToolReport, ...], *, writer: str
) -> tuple[ToolReport, ...]:
    """Install what a node is missing and this package can supply, and verify it.

    AN INSTALL THAT RAN IS NOT AN INSTALL THAT WORKED, so the node is probed
    again and every tool this run installed must answer present. When the
    install script fails, or exits 0 with a tool still absent, what it did
    land is removed (:func:`rollback_install`) before the failure is
    raised: a half-installed node is worse than an untouched one, because
    it looks closer to ready than it is, and these are other people's
    machines, left as they were found (``state-change-verified``, board
    task cc7222ca).

    Args:
        node: The node to install on.
        reports: What it answered when probed.
        writer: Who is asking, which names the probe scripts on the node
            (:func:`fleet.core.toolchain.attempt_toolchain`).

    Returns:
        What the node answered after the install, every installed tool
        present; ``reports`` itself when there was nothing to install, which
        is a real answer rather than an error -- a node may be missing only
        things that have no automatic install.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` when the
            install script fails, carrying the node's own stderr; with
            ``DISPATCH_FAILED`` when it exits 0 and a tool it installed is
            still absent. Either message names what the rollback removed. A
            probe or a rollback that fails raises its own error instead.
    """
    tools = installable(reports)
    if not tools:
        return reports
    managers = available_managers(reports)
    spoken = dialect.for_platform(node["platform"])
    outcome = remote.attempt_script(
        node["host"],
        spoken.script_path(node["stage_root"], names.INSTALL_STEM),
        install_script(tools, managers, platform=node["platform"]),
        platform=node["platform"],
    )
    failure = outcome["failure"]
    if failure is not None:
        removed = rollback_install(node, tools, managers, writer=writer)
        raise AppError(failure["code"], f"{failure['message']}; {_rolled_back(removed)}")
    after = probe_toolchain(node, writer=writer)
    absent = tuple(name for name in tools if name not in _present_names(after))
    if absent:
        removed = rollback_install(node, tools, managers, writer=writer)
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"{node['host']}: installing {', '.join(tools)} exited 0 but the probe still "
            f"finds {', '.join(absent)} absent; {_rolled_back(removed)}",
        )
    return after


def rollback_install(
    node: NodeConfig, tools: tuple[str, ...], managers: tuple[str, ...], *, writer: str
) -> tuple[str, ...]:
    """Remove the tools an unfinished install landed, and verify they are gone.

    Every tool :func:`install_missing` names was absent before it ran, so
    one that the probe now finds present is one this install put there.

    Args:
        node: The node the install ran on.
        tools: The tools the install was for.
        managers: The managers the install was rendered with, so each tool
            is removed by the manager that installed it.
        writer: Who is asking, for the probe scripts' names.

    Returns:
        The tools removed, in the install's order; empty when none had
        landed, in which case nothing is sent to the node.

    Raises:
        AppError: With ``NODE_UNREACHABLE`` or ``DISPATCH_FAILED`` from a
            probe or the removal script, or ``DISPATCH_FAILED`` when a tool
            it removed still answers present, naming it, since the node then
            holds part of a toolchain nobody asked to keep.
    """
    present = _present_names(probe_toolchain(node, writer=writer))
    landed = tuple(name for name in tools if name in present)
    if not landed:
        return ()
    spoken = dialect.for_platform(node["platform"])
    remote.run_script(
        node["host"],
        spoken.script_path(node["stage_root"], names.UNINSTALL_STEM),
        uninstall_script(landed, managers, platform=node["platform"]),
        platform=node["platform"],
    )
    remaining = _present_names(probe_toolchain(node, writer=writer))
    stuck = tuple(name for name in landed if name in remaining)
    if stuck:
        raise AppError(
            FleetErrorCode.DISPATCH_FAILED,
            f"{node['host']}: rolling back an unfinished install left {', '.join(stuck)} "
            "installed; remove by hand: "
            + "; ".join(uninstall_command(name, managers) for name in stuck),
        )
    return landed


def _present_names(reports: tuple[ToolReport, ...]) -> frozenset[str]:
    """Name the tools a probe found present.

    Args:
        reports: What a node answered.

    Returns:
        The names of the present ones.
    """
    return frozenset(report["name"] for report in reports if report["present"])


def _rolled_back(removed: tuple[str, ...]) -> str:
    """Say what a rollback removed, for the end of a failure's message.

    Args:
        removed: What :func:`rollback_install` returned.

    Returns:
        The clause naming the removed tools, or saying none had landed.
    """
    if not removed:
        return "rolled back nothing: none of them had landed"
    return f"rolled back {', '.join(removed)}"


__all__ = [
    "install_missing",
    "install_script",
    "installable",
    "rollback_install",
    "uninstall_script",
]
