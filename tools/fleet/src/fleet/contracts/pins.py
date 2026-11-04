"""The workspace's checks on a project that names its node (MCPs board tasks bf7a9460, f19d66a5).

A project may name the one node it runs on (``ProjectConfig.node``), for a
suite that reads what only that host holds: MCPs' served lanes compare what
one host's own search container serves on 127.0.0.1:8090 with the commit's
files. :func:`fleet.core.capacity.assess` refuses the project on every other
node by matching the pin against the weighing node's host, so the name must
be a node the workspace declares; anything else would be refused by every
node and wait for one that never claims it. A node's host is its machine
file's name (:mod:`fleet.contracts.machine`), so a declared pin always
matches its host.

THE HUB SUITE IS PINNED TO THE ROSTER'S HUB. ``tools/fleet-execution-hub``
reads every node through the hub's own ssh and the hub's records (MCPs board
task 3d9eec0d), so it means something only on the machine whose role is
``hub``. API 96a73d084 (board task 927bbd25) gave that role to lacrymal while
the suite's row still named diphtheria, and on 2026-11-04 fleet job d2088f5d
ran it there: diphtheria's own ssh could not reach diphtheria, and a
test_execution_projects case pinned the literal (board task f19d66a5). So
:func:`check_hub_suite` refuses a hub suite whose node is not the one hub,
and the next hub cutover fails when the registry loads instead of at a check
on the old host.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import JSONTypeError

from fleet.contracts.machine import Machine, MachineRole
from fleet.contracts.node import NodeConfig
from fleet.contracts.project import ProjectConfig

#: The project whose cases read every node from the hub.
HUB_SUITE: Final = "tools/fleet-execution-hub"


def check_pinned_nodes(
    projects: dict[str, ProjectConfig], nodes: dict[str, NodeConfig]
) -> dict[str, ProjectConfig]:
    """Refuse a project pinned to a node the workspace cannot match.

    Args:
        projects: The decoded projects.
        nodes: The decoded nodes.

    Returns:
        ``projects``, unchanged.

    Raises:
        JSONTypeError: Naming the project and its pin when the pin is not a
            declared node.
    """
    for path, project in sorted(projects.items()):
        pinned = project["node"]
        if pinned is None:
            continue
        if pinned not in nodes:
            raise JSONTypeError(
                f"project {path} is pinned to node {pinned!r}, which the workspace does not "
                f"declare (it declares {', '.join(sorted(nodes))}); no runner would ever claim it"
            )
    return projects


def check_hub_suite(
    projects: dict[str, ProjectConfig], machines: dict[str, Machine]
) -> dict[str, ProjectConfig]:
    """Refuse a hub suite pinned anywhere but the one machine whose role is hub.

    A workspace that declares no hub suite passes: the rule is about where
    the suite runs, not whether one is declared.

    Args:
        projects: The decoded projects.
        machines: The decoded machines.

    Returns:
        ``projects``, unchanged.

    Raises:
        JSONTypeError: Naming the suite's pin and the roster's hubs when the
            pin is not the one hub, the suite is unpinned, or the roster
            declares no hub or more than one.
    """
    suite = projects.get(HUB_SUITE)
    if suite is None:
        return projects
    hubs = sorted(name for name, machine in machines.items() if machine["role"] is MachineRole.HUB)
    if suite["node"] is None or [suite["node"]] != hubs:
        named = ", ".join(hubs) if hubs else "no machine"
        raise JSONTypeError(
            f"project {HUB_SUITE} is pinned to node {suite['node']!r}, and the roster's hub is "
            f"{named}; its cases read every node through the hub's own ssh and records, so it "
            "is pinned to the one machine whose role is hub"
        )
    return projects


__all__ = ["HUB_SUITE", "check_hub_suite", "check_pinned_nodes"]
