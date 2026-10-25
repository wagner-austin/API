"""The projects whose code holds no Windows code run on Linux (MCPs board task 6c89be43).

services/Model-Trainer required gpu and windows, and libs/cleargbm_rs and
libs/covenant_nn windows and rust, unchanged since the registry split (API
7769257b9). On 2026-10-24 none of the three directories held a .ps1, .bat or
.cmd file, git grep for powershell, pwsh, win32, sys.platform, cfg(windows)
or .exe found nothing but poetry.lock platform markers, and each Makefile
includes scripts/make/shell.mk. loki, the one Windows node, carries neither a
GPU nor rust, so no node could run any of them: three Model-Trainer checks
had sat queued since 2026-10-14 and covenant_ml and cleargbm_rs checks since
2026-10-22, every runner answering lacks-tags. The operator, that day: "bro
we removed the eindoes code from model trainer why rhe fu l doesit need to
run on windows??? same for rust what rhe fuck ??"

Each project is exercised against the shipped registry: it requires no
windows tag, and an enabled Linux node, idle with memory to spare, grants it
workers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleet.contracts.node import LiveLoad, NodeConfig, NodeState
from fleet.contracts.tags import NodeTag, node_tags
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core.capacity import assess
from fleet.core.workspace_files import read_workspace_document

#: Each project and the enabled Linux nodes that grant it workers, measured
#: against the shipped registry on 2026-10-25 (diphtheria carries the tags
#: too and refuses, because it checks only the projects pinned to it).
LINUX_GRANTS = {
    "libs/cleargbm_rs": ("lacrymal", "serendipity"),
    "libs/covenant_nn": ("lacrymal", "serendipity"),
    "services/Model-Trainer": ("demure", "lavender"),
}


def _shipped() -> FleetWorkspace:
    """The registry this package ships.

    Returns:
        The decoded workspace.
    """
    return decode_fleet_workspace(
        read_workspace_document(Path(__file__).resolve().parents[1] / "registry")
    )


def _idle(node: NodeConfig) -> NodeState:
    """A node with nothing of the fleet's on it and memory to spare.

    Args:
        node: The node's declaration.

    Returns:
        Its state with 24 GB free, no live run, no reservation and no
        pressure read.
    """
    return NodeState(
        host=node["host"],
        free_ram_gb=24.0,
        free_disk_gb=100.0,
        live=LiveLoad(runs=0, workers=0, ram_gb=0.0),
        ci_slice=None,
        reserved_cores=0,
        action_cache=None,
        cpu_pressure_pct=None,
        io_pressure_pct=None,
    )


@pytest.mark.parametrize(("project_name", "granting"), sorted(LINUX_GRANTS.items()))
def test_the_project_requires_no_windows_and_enabled_linux_nodes_grant_it(
    project_name: str, granting: tuple[str, ...]
) -> None:
    workspace = _shipped()
    project = workspace["projects"][project_name]

    assert NodeTag.WINDOWS not in project["required_tags"]
    granted = tuple(
        name
        for name, node in sorted(workspace["nodes"].items())
        if node["platform"] == "linux"
        and node["enabled"]
        and assess(node, _idle(node), project, node_tags(node))["workers"] > 0
    )
    assert granted == granting
