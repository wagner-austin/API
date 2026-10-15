"""The workspace's check on a project that names its node (MCPs board task bf7a9460).

A project may name the one node it runs on (``ProjectConfig.node``), for a
suite that reads what only that host holds: MCPs' served lanes compare what
one host's own search container serves on 127.0.0.1:8090 with the commit's
files. :func:`fleet.core.capacity.assess` refuses the project on every other
node by matching the pin against the weighing node's host, so the name must
be a node the workspace declares, whose host is that same name; anything
else would be refused by every node and wait for one that never claims it.
"""

from __future__ import annotations

from platform_core.json_utils import JSONTypeError

from fleet.contracts.node import NodeConfig
from fleet.contracts.project import ProjectConfig


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
            declared node, or names a node whose host is another name.
    """
    for path, project in sorted(projects.items()):
        pinned = project["node"]
        if pinned is None:
            continue
        declared = nodes.get(pinned)
        if declared is None:
            raise JSONTypeError(
                f"project {path} is pinned to node {pinned!r}, which the workspace does not "
                f"declare (it declares {', '.join(sorted(nodes))}); no runner would ever claim it"
            )
        if declared["host"] != pinned:
            raise JSONTypeError(
                f"project {path} is pinned to node {pinned!r}, whose host is "
                f"{declared['host']!r}; a pin is matched against the weighing node's host, so "
                "it must name a node whose host is its name"
            )
    return projects


__all__ = ["check_pinned_nodes"]
