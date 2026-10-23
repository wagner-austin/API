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

from platform_core.json_utils import dump_json_str

from fleet.cli import agent as agent_cli
from fleet.core import _test_hooks, tick_run
from tests._queue_fakes import queue_env

#: The machine a hub tick runs on, as the hostname hook reports it: a hub
#: lane job is reported started there, wherever the control duty is held
#: (MCPs board task c39ee056).
HUB_HOST = "diphtheria"

#: The unit every agent tick of these suites runs in, as the hub runner
#: names a tick's unit (MCPs board task 05dd6588).
TICK_UNIT = "session-audit-job-0123456789abcdef0123456789abcdef.service"

#: What such a tick reads from ``/proc/self/cgroup``, measured in that shape
#: on diphtheria on 2026-10-21.
TICK_CGROUP = f"0::/user.slice/user-1000.slice/user@1000.service/app.slice/{TICK_UNIT}\n"

#: The queue's answer to the claim pass's first call, the jobs held under
#: the runner's label, when it holds none.
NOTHING_HELD = dump_json_str({"jobs": []})

#: The tools a tick asks before any report: what it holds, then its claim.
HELD_THEN_CLAIMED = ("dispatch_list", "dispatch_claim")


def tick_run_id(machine: str = HUB_HOST) -> str:
    """The run id a job claimed by a tick of these suites is started as.

    Args:
        machine: The hostname the tick ran on.

    Returns:
        ``<TICK_UNIT> on <machine>``.
    """
    return f"{TICK_UNIT} on {machine}"


def _read_in_tick_unit(path: pathlib.Path) -> str:
    """Read a file as a process in :data:`TICK_UNIT` would.

    Args:
        path: The file.

    Returns:
        :data:`TICK_CGROUP` for ``/proc/self/cgroup``, the file's real text
        for any other.
    """
    if path == tick_run.PROC_SELF_CGROUP:
        return TICK_CGROUP
    return _test_hooks._default_read_text(path)


def _exists_in_tick_unit(path: pathlib.Path) -> bool:
    """Say whether a file exists, as on a machine with ``/proc``.

    Args:
        path: The file.

    Returns:
        True for ``/proc/self/cgroup``, the real answer for any other.
    """
    return path == tick_run.PROC_SELF_CGROUP or _test_hooks._default_file_exists(path)


def hub_host() -> str:
    """Report :data:`HUB_HOST`, bound as the hostname hook.

    Returns:
        ``diphtheria``.
    """
    return HUB_HOST


def bind_hub_tick() -> None:
    """Bind the hooks so this process runs as a hub runner's tick.

    The queue's two variables are set, the hostname is :data:`HUB_HOST`, and
    ``/proc/self/cgroup`` reads :data:`TICK_CGROUP`: the suite's own process
    runs in whatever control group launched it, which differs between a
    fleet node's unit, CI and a login shell, so an agent tick here reads the
    one its test names.
    """
    _test_hooks.env = queue_env()
    _test_hooks.hostname = hub_host
    _test_hooks.read_text = _read_in_tick_unit
    _test_hooks.file_exists = _exists_in_tick_unit


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


__all__ = [
    "HELD_THEN_CLAIMED",
    "HUB_HOST",
    "NOTHING_HELD",
    "TICK_CGROUP",
    "TICK_UNIT",
    "agent_argv",
    "bind_hub_tick",
    "hub_host",
    "netboot_argv",
    "sessions_argv",
    "tick_run_id",
]
