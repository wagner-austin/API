"""The ``fleet-agent`` command lines the agent suites run a tick with.

Shared by the agent suites (``test_agent.py``, ``test_agent_rebuild.py``,
``test_agent_restart.py``, ``test_agent_netboot.py`` and
``test_declared.py``, split by role at the
file-size ceiling) so each runner's identity stays one set of literals. Out
of ``conftest.py`` since the sessions lane (MCPs board task c39ee056) gave
the agent a second runner, which took that file past the ceiling.
"""

from __future__ import annotations

import pathlib

from fleet.cli import agent as agent_cli

#: The machine a hub tick runs on, as the hostname hook reports it: a hub
#: lane job is reported started there, wherever the control duty is held
#: (MCPs board task c39ee056).
HUB_HOST = "austinpc"


def hub_host() -> str:
    """Report :data:`HUB_HOST`, bound as the hostname hook.

    Returns:
        ``austinpc``.
    """
    return HUB_HOST


def agent_argv(config_path: pathlib.Path, repo: pathlib.Path) -> list[str]:
    """Build the agent CLI's argument list for one hub tick.

    Args:
        config_path: The workspace document.
        repo: The monorepo root on this machine.

    Returns:
        The argument list.
    """
    return [
        "--config",
        str(config_path),
        agent_cli.AGENT_FLAG,
        f"fleet-hub-{HUB_HOST}",
        agent_cli.SESSION_FLAG,
        "33333333-cccc-4ccc-8ccc-333333333333",
        agent_cli.ROOT_FLAG,
        str(repo),
        agent_cli.LANE_FLAG,
        "hub",
    ]


def sessions_argv(config_path: pathlib.Path, repo: pathlib.Path, node: str) -> list[str]:
    """Build the agent CLI's argument list for one sessions tick on ``node``.

    Args:
        config_path: The workspace document.
        repo: The monorepo root on this machine.
        node: The machine holding the sessions duty.

    Returns:
        The argument list, with the sessions runner's identity.
    """
    return [
        "--config",
        str(config_path),
        agent_cli.AGENT_FLAG,
        f"fleet-sessions-{node}",
        agent_cli.SESSION_FLAG,
        "44444444-dddd-4ddd-8ddd-444444444444",
        agent_cli.ROOT_FLAG,
        str(repo),
        agent_cli.LANE_FLAG,
        "sessions",
        agent_cli.NODE_FLAG,
        node,
    ]


def netboot_argv(
    config_path: pathlib.Path, repo: pathlib.Path, place: dict[str, pathlib.Path]
) -> list[str]:
    """Build the agent CLI's argument list for one netboot tick.

    Args:
        config_path: The workspace document.
        repo: The monorepo root on this machine.
        place: The release, served root and env file, keyed by flag; the
            node binary is always ``node``.

    Returns:
        The argument list, with the netboot runner's identity.
    """
    return [
        "--config",
        str(config_path),
        agent_cli.AGENT_FLAG,
        f"fleet-netboot-{HUB_HOST}",
        agent_cli.SESSION_FLAG,
        "55555555-eeee-4eee-8eee-555555555555",
        agent_cli.ROOT_FLAG,
        str(repo),
        agent_cli.LANE_FLAG,
        "netboot",
        agent_cli.NODEJS_FLAG,
        "node",
        *(token for flag, path in place.items() for token in (flag, str(path))),
    ]


__all__ = ["HUB_HOST", "agent_argv", "hub_host", "netboot_argv", "sessions_argv"]
