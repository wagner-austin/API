"""No shipped project needs Windows, and every one an enabled node grants (board task e705f08c).

loki was the fleet's one Windows node, at one check at a time, and every
project that required windows, or elevated, which only loki carried, queued
for that one laptop. On 2026-10-25 the operator ruled "dude. we cannot have
tge bottlnecks. so no windows shit then ok ?", and loki left the roster to
return once it runs NixOS. Before that, MCPs board tasks 6c89be43 and
e35812cc had found that ten projects requiring windows held no Windows code
at all (:data:`LINUX_PROJECTS`).

So this file pins three things against the shipped registry: no project file
names windows or elevated in its ``required_tags`` (read from the raw files,
since the decoder refuses either word, so a row naming one fails here by
name before it fails every decode), no machine file declares a platform but
linux, loki included since it returned as a linux vpn-jump for its NixOS
install (MCPs board task 7d7ea0e9), and every project that ever required
windows (:data:`FORMERLY_WINDOWS` and :data:`LINUX_PROJECTS`) is granted
workers by an enabled node, idle with memory to spare. A tool tag (ffmpeg,
chrome, hooks, go, pdftotext) is one no declaration holds; a runner adds it
on the tick its probe finds the tool, so it is counted here as present and
the project's fleet check is what shows a node has it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue, require_dict, require_list

from fleet.contracts.node import LiveLoad, NodeConfig, NodeState
from fleet.contracts.tags import TOOL_TAG, node_tags
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core.capacity import assess
from fleet.core.workspace_files import read_workspace_document

#: The registry this package ships.
REGISTRY = Path(__file__).resolve().parents[1] / "registry"

#: The projects that required windows until board task e705f08c. Whether
#: each one's check passes on Linux is its own fleet check's answer, and a
#: Windows dependency it finds is removed from that project, never answered
#: by giving the tag back.
FORMERLY_WINDOWS = (
    "chat",
    "chat/execution",
    "clients/NavProbe",
    "clients/TankpitBot-execution",
    "corvis-stick",
    "libs/instrument_io",
    "services/procart-api",
    "slime",
    "tree-bot",
)

#: The projects whose windows tag was dropped because nothing in them needed
#: it (MCPs board tasks 6c89be43 and e35812cc).
LINUX_PROJECTS = (
    "Dashboards",
    "Dashboards/rabbit",
    "LSTM",
    "clients/TankpitBot/web",
    "idle",
    "libs/cleargbm_rs",
    "libs/covenant_nn",
    "metabolomics-dashboard",
    "services/Model-Trainer",
    "services/grandma-api",
)

#: Every project the grant case reads, both groups above.
GRANTED_PROJECTS: tuple[str, ...] = (*FORMERLY_WINDOWS, *LINUX_PROJECTS)

#: The words no row may require: no node carries either.
RETIRED_TAGS = ("windows", "elevated")


def _shipped() -> FleetWorkspace:
    """The registry this package ships, decoded.

    Returns:
        The decoded workspace.
    """
    return decode_fleet_workspace(read_workspace_document(REGISTRY))


def _idle(node: NodeConfig) -> NodeState:
    """A node with nothing of the fleet's on it and memory to spare.

    Args:
        node: The node's declaration.

    Returns:
        Its state with 24 GB free, no live run, no reservation and no
        pressure.
    """
    return NodeState(
        host=node["host"],
        free_ram_gb=24.0,
        free_disk_gb=100.0,
        live=LiveLoad(runs=0, workers=0, ram_gb=0.0),
        ci_slice=None,
        reserved_cores=0,
        action_cache=None,
        cpu_pressure_pct=0.0,
        io_pressure_pct=0.0,
    )


def _required(row: JSONValue) -> list[JSONValue]:
    """One raw project row's ``required_tags``.

    Args:
        row: The row as its file holds it.

    Returns:
        The list.
    """
    return require_list(require_dict({"row": row}, "row"), "required_tags")


def test_no_project_row_requires_windows_or_elevated() -> None:
    rows = require_dict(read_workspace_document(REGISTRY), "projects")
    naming = sorted(
        f"{name} requires {tag}"
        for name, row in rows.items()
        for tag in RETIRED_TAGS
        if tag in _required(row)
    )

    assert len(rows) > 100, f"read {len(rows)} project rows; the registry ships more"
    assert naming == []


def test_every_machine_is_linux_and_loki_returned_as_linux() -> None:
    machines = require_dict(read_workspace_document(REGISTRY), "machines")
    platforms = {name: require_dict(machines, name)["platform"] for name in machines}

    assert platforms["loki"] == "linux"
    assert {name for name, platform in platforms.items() if platform != "linux"} == set()


@pytest.mark.parametrize("project_name", GRANTED_PROJECTS)
def test_an_enabled_node_grants_every_project_that_once_needed_windows(
    project_name: str,
) -> None:
    workspace = _shipped()
    project = workspace["projects"][project_name]
    detected = frozenset(project["required_tags"]) & frozenset(TOOL_TAG.values())

    granted = [
        name
        for name, node in sorted(workspace["nodes"].items())
        if node["enabled"]
        and assess(node, _idle(node), project, node_tags(node) | detected)["workers"] > 0
    ]
    assert granted, f"no enabled node grants {project_name} workers"
