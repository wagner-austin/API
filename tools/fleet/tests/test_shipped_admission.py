"""The production registry admits one check at a time where two were measured over budget.

A check runs wider than its test workers (lint, type checks, a build), so on
these nodes a second check fits the core pool and still pushes both past the
300 s budget. Each node's ``checks_at_once`` was set after that was measured:

- serendipity, MCPs board task a85ef09e: two checks inside the pool's 4 cores
  pushed MCPs/sms-gateway from 162 s to 306 s.
- loki, MCPs board task f6c2451a: MCPs/packages/session-audit (350 s) and
  corvis-stick both crossed the budget, run side by side on 2026-10-09.
- diphtheria, MCPs board task 18816b8f: the shared-library suite took 188 s
  alone and 410 s beside another check.
- lavender-wsl, MCPs board task 30dba6d7: MCPs/mcp-shared took 130 s alone
  (fleet job f55b38a0) and 305 s beside MCPs/execution-deploy and
  MCPs/packages/maketools (fleet job aad491ff) on 2026-10-16.

Admission is exercised against the shipped registry, so removing a node's
bound restores the second admission that was measured, and a node gaining or
losing one is a change this file has to be told about.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad, NodeState
from fleet.contracts.tags import node_tags
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core.capacity import assess
from fleet.core.workspace_files import read_workspace_document

#: Each bounded node and the workers it grants MCPs/mcp-shared alone: half
#: the cores its owner leaves, rounded up, with memory to spare at 24 GB free.
BOUNDED_GRANTS = {"diphtheria": 2, "lavender-wsl": 4, "loki": 6, "serendipity": 2}


def _shipped() -> FleetWorkspace:
    """The registry this package ships.

    Returns:
        The decoded workspace.
    """
    return decode_fleet_workspace(
        read_workspace_document(Path(__file__).resolve().parents[1] / "registry")
    )


def test_the_measured_nodes_are_the_bounded_ones() -> None:
    bounds = {
        name: node["budget"]["checks_at_once"]
        for name, node in _shipped()["nodes"].items()
        if node["budget"]["checks_at_once"] is not None
    }

    assert bounds == {"diphtheria": 1, "lavender-wsl": 1, "loki": 1, "serendipity": 1}


@pytest.mark.parametrize(("name", "grant"), sorted(BOUNDED_GRANTS.items()))
def test_a_bounded_node_waits_for_the_whole_check_to_end(name: str, grant: int) -> None:
    workspace = _shipped()
    node = workspace["nodes"][name]
    project = workspace["projects"]["MCPs/mcp-shared"]
    state = NodeState(
        host=node["host"],
        free_ram_gb=24.0,
        free_disk_gb=100.0,
        live=LiveLoad(runs=0, workers=0, ram_gb=0.0),
        ci_slice=None,
        reserved_cores=0,
        action_cache=None,
        cpu_pressure_pct=None,
    )
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
