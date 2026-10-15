"""The production registry prevents the measured two-check CPU contention.

MCPs board task 18816b8f: the same shared-library suite took 188 seconds
alone and 410 seconds while cloudflared held another two workers. Both
grants fit the four-core pool, but the complete checks run wider than
their test workers. Exercise admission against the shipped registry so
removing its concurrency bound restores the offending second admission.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad, NodeState
from fleet.contracts.tags import node_tags
from fleet.contracts.workspace import decode_fleet_workspace
from fleet.core.capacity import assess
from fleet.core.workspace_files import read_workspace_document


def test_diphtheria_waits_for_the_whole_check_to_end() -> None:
    workspace = decode_fleet_workspace(
        read_workspace_document(Path(__file__).resolve().parents[1] / "registry")
    )
    node = workspace["nodes"]["diphtheria"]
    project = workspace["projects"]["MCPs/mcp-shared"]
    state = NodeState(
        host=node["host"],
        free_ram_gb=24.0,
        free_disk_gb=100.0,
        live=LiveLoad(runs=0, workers=0, ram_gb=0.0),
        ci_slice=None,
        reserved_cores=0,
    )
    assert assess(node, state, project, node_tags(node)) == {
        "workers": 2,
        "code": None,
        "reason": "",
    }

    state["live"] = LiveLoad(runs=1, workers=2, ram_gb=1.0)
    refused = assess(node, state, project, node_tags(node))
    assert refused["workers"] == 0
    assert refused["code"] == FleetErrorCode.NODE_OWNER_RESERVED
    assert "1 budgeted check(s) at once" in refused["reason"]
    assert "1 are live or launching" in refused["reason"]

    state["live"] = LiveLoad(runs=0, workers=0, ram_gb=0.0)
    assert assess(node, state, project, node_tags(node)) == {
        "workers": 2,
        "code": None,
        "reason": "",
    }
