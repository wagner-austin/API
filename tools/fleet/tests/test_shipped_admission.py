"""The production registry admits one check at a time where two were measured over budget.

A check runs wider than its test workers (lint, type checks, a build), so on
these nodes a second check fits the core pool and still pushes both past the
300 s budget. Each node's ``checks_at_once`` was set after that was measured:

- loki, MCPs board task f6c2451a: MCPs/packages/session-audit (350 s) and
  corvis-stick both crossed the budget, run side by side on 2026-10-09.
- diphtheria, MCPs board task 18816b8f: the shared-library suite took 188 s
  alone and 410 s beside another check.
- serendipity, MCPs board task a85ef09e: two checks inside the pool's 4 cores
  pushed MCPs/sms-gateway from 162 s to 306 s on Windows. Its reimage to
  Ubuntu on 2026-10-20 (MCPs board task ac5e8a88) kept the same 4 physical
  cores, so the Linux worker it was provisioned as (MCPs board task 632b0802)
  keeps the bound until a check measured there says otherwise.
- lenovold, MCPs board task 6747753b: no second check was run there. It was
  provisioned as a Linux worker on 2026-10-21 with 2 physical cores (an
  i5-5200U, 4 threads), so its pool of 2 is half of serendipity's 4, where
  two checks were measured over budget. It takes one check at a time from its
  first dispatch, and that holds until a check measured there says otherwise.
  Its dispatch is held as well: its first job, MCPs/sms-gateway (fleet job
  1b910743), took 307 s of 300 alone. The bound stays pinned for the day the
  hold is lifted.
lavender-wsl carried a fourth bound (MCPs board task 30dba6d7: MCPs/mcp-shared
took 130 s alone and 305 s beside two other checks on 2026-10-16) until it
went with lavender's reimage to Ubuntu on 2026-10-19 (MCPs board task
1e9ab738).

diphtheria also checks only the projects pinned to it (MCPs board task
d577036f): one check at a time, it still ran 10 of its 341 checks in the 7
days to 2026-10-18 past the budget with their tests passing, beside the
production stack, against 3 of about 1,500 on the other nodes.

Admission is exercised against the shipped registry, so removing a node's
bound restores the second admission that was measured, and a node gaining or
losing one is a change this file has to be told about.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad, NodeConfig, NodeState
from fleet.contracts.tags import node_tags
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core.capacity import assess
from fleet.core.workspace_files import read_workspace_document

#: Each bounded node, a project it checks, and the workers it grants that
#: project alone: half the cores its owner leaves, rounded up, with memory to
#: spare at 24 GB free. diphtheria is weighed with a suite pinned to it,
#: because it checks no other. loki and serendipity leave their owner only
#: the operating system's 2 cores since MCPs board task e447de72 (the
#: operator, 2026-10-19: "i dont uze amt kf rhe pcs perosnally rhwyre all
#: workers"), and loki a core more for each of the four GitHub runners
#: runners.json declares on it (MCPs board task 1e9ab738), so 10 and 6
#: cores are left to the fleet. lenovold keeps the same 2 of its 4, and the
#: suite's declared minimum of 2 workers, all the cores left, outranks half
#: of them (MCPs board task 6747753b).
BOUNDED_GRANTS = {
    "diphtheria": ("MCPs/doc-extract-api", 2),
    "lenovold": ("MCPs/mcp-shared", 2),
    "loki": ("MCPs/mcp-shared", 5),
    "serendipity": ("MCPs/mcp-shared", 3),
}

#: The suites that need diphtheria's production stack or read what only it
#: holds, each pinned there, so it checks them and nothing else.
DIPHTHERIA_PINNED = (
    "MCPs/doc-extract-api",
    "MCPs/execution-netboot-diphtheria",
    "MCPs/execution-served-diphtheria",
    "MCPs/pg-backup-sidecar",
    "MCPs/phone-device-execution",
    "MCPs/phone-release-execution",
    "MCPs/transcriber-api",
    "tools/fleet-execution-hub",
)


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


def test_the_measured_nodes_are_the_bounded_ones() -> None:
    bounds = {
        name: node["budget"]["checks_at_once"]
        for name, node in _shipped()["nodes"].items()
        if node["budget"]["checks_at_once"] is not None
    }

    assert bounds == {"diphtheria": 1, "lenovold": 1, "loki": 1, "serendipity": 1}


@pytest.mark.parametrize(("name", "measured"), sorted(BOUNDED_GRANTS.items()))
def test_a_bounded_node_waits_for_the_whole_check_to_end(
    name: str, measured: tuple[str, int]
) -> None:
    project_name, grant = measured
    workspace = _shipped()
    node = workspace["nodes"][name]
    project = workspace["projects"][project_name]
    state = _idle(node)
    assert assess(node, state, project, node_tags(node)) == {
        "workers": grant,
        "code": None,
        "reason": "",
    }

    state["live"] = LiveLoad(runs=1, workers=grant, ram_gb=1.0)
    refused = assess(node, state, project, node_tags(node))
    assert refused["workers"] == 0
    assert refused["code"] == FleetErrorCode.NODE_OWNER_RESERVED
    assert "1 budgeted check(s) at once" in refused["reason"]
    assert "1 are live or launching" in refused["reason"]

    state["live"] = LiveLoad(runs=0, workers=0, ram_gb=0.0)
    assert assess(node, state, project, node_tags(node)) == {
        "workers": grant,
        "code": None,
        "reason": "",
    }


def test_only_diphtheria_checks_only_its_pinned_projects() -> None:
    declared = {
        name for name, node in _shipped()["nodes"].items() if node["budget"]["pinned_checks_only"]
    }

    assert declared == {"diphtheria"}


def test_diphtheria_checks_the_projects_pinned_to_it_and_no_other() -> None:
    workspace = _shipped()
    node = workspace["nodes"]["diphtheria"]
    taken = {
        name
        for name, project in workspace["projects"].items()
        if assess(node, _idle(node), project, node_tags(node))["code"] is None
    }

    assert taken == set(DIPHTHERIA_PINNED)
    for suite in ("MCPs/watcher", "MCPs/mcp-shared", "MCPs/packages/db"):
        refused = assess(node, _idle(node), workspace["projects"][suite], node_tags(node))
        assert refused["code"] == FleetErrorCode.NODE_NOT_PINNED
