"""Whether a node's declared memory adds up: CI's cap, the reservation and one job.

MCPs board task 5d6e57e7. On 2026-09-29 lavender's runners.slice (45a4f22b)
was allowed 18 GB (``MemoryMax``, 16 GB ``MemoryHigh``) of the WSL VM that
lavender-wsl runs in, while fleet.json reserved 12 GB of that same VM before
any fleet worker. The node declares 25.4 GB, so while CI held its budget the
lane could never dispatch: 18 + 12 is past the machine before a single
worker. Every tick answered ``NODE_OWNER_RESERVED`` from 17:15Z, and the
testdb queue stood still with twelve rows, by declaration rather than by
load.

The two numbers live in two files (``runners.json`` and ``fleet.json``) and
were each reasonable alone. This is the check that reads both: for every
node, the CI slice its host may run inside the node (none unless the node's
``wsl_host`` is a runners.json host with a slice), plus the node's
``reserved_ram_gb``, plus the smallest job the node can serve (a project's
``minimum_workers`` times its ``worker_ram_gb``, over the projects whose
tags the node carries), must be at most the node's declared ``ram_gb``.
"""

from __future__ import annotations

from fleet.contracts.node import NodeConfig
from fleet.contracts.project import ProjectConfig
from fleet.contracts.runners import RunnerSpec
from fleet.contracts.tags import missing_tags
from fleet.contracts.workspace import FleetWorkspace


def slice_cap_gb(node: NodeConfig, runners: RunnerSpec) -> int:
    """The CI slice's ``MemoryMax`` inside a node, or 0 where there is none.

    Args:
        node: The node's declaration.
        runners: The runner roster.

    Returns:
        ``memory_max_gb`` of the runners.json host the node's ``wsl_host``
        names; 0 for a node with no ``wsl_host`` or one whose host runs no
        runners here.
    """
    for host in runners["hosts"]:
        if node["wsl_host"] == host["host"]:
            return host["ci_slice"]["memory_max_gb"]
    return 0


def smallest_job_gb(node: NodeConfig, projects: tuple[ProjectConfig, ...]) -> float:
    """The memory of the smallest dispatch a node could take.

    Args:
        node: The node's declaration.
        projects: Every registered project.

    Returns:
        The least ``minimum_workers * worker_ram_gb`` over the projects whose
        required tags the node carries; 0.0 when it can serve none, since
        such a node claims nothing and so needs no room.
    """
    sizes = [
        project["minimum_workers"] * project["worker_ram_gb"]
        for project in projects
        if not missing_tags(node, project["required_tags"])
    ]
    return min(sizes, default=0.0)


def budget_gaps(workspace: FleetWorkspace, runners: RunnerSpec) -> tuple[str, ...]:
    """Every node whose declarations cannot all hold at once.

    Args:
        workspace: The fleet workspace.
        runners: The runner roster.

    Returns:
        One sentence per node whose CI slice, reservation and smallest job
        exceed its ``ram_gb``, naming all four numbers; empty when every node
        adds up.
    """
    projects = tuple(workspace["projects"].values())
    gaps: list[str] = []
    for name, node in workspace["nodes"].items():
        cap = slice_cap_gb(node, runners)
        reserved = node["budget"]["reserved_ram_gb"]
        job = smallest_job_gb(node, projects)
        if cap + reserved + job > node["ram_gb"]:
            gaps.append(
                f"{name}: CI slice {cap} GB + reservation {reserved:.1f} GB + smallest job "
                f"{job:.2f} GB is {cap + reserved + job:.2f} GB, past its {node['ram_gb']:.1f} "
                "GB, so while CI holds its cap the node can never dispatch; lower the "
                "reservation in fleet.json or the slice in runners.json"
            )
    return tuple(gaps)


__all__ = ["budget_gaps", "slice_cap_gb", "smallest_job_gb"]
