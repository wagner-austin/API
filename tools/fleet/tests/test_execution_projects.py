"""The fleet runs tools/fleet's host cases once per platform (board task 465689f5).

API's deploy refuses a commit unless the fleet queue records a passed run of
tools/fleet-execution and of tools/fleet-execution-linux at it (the root
Makefile's ``executed``, pinned in tools/maketools' execution-gate wiring
test). Those verdicts mean both halves ran only if the registry sends each
project to a node of its platform and both stage the one directory whose
check runs the host cases, which is what this decodes the real registry to
require.

The Linux half also requires ``docker``: its host cases include one that runs
a docker project's build as the node's execdocker user against that user's
rootless daemon (MCPs board task a8ee9b21), which only a node carrying the
tag has.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str

from fleet.contracts.source import ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project


@pytest.mark.parametrize(
    ("name", "tags"),
    [
        ("tools/fleet-execution", (NodeTag.WINDOWS,)),
        ("tools/fleet-execution-linux", (NodeTag.LINUX, NodeTag.DOCKER)),
    ],
)
def test_each_half_runs_the_execution_directory_on_its_platform(
    name: str, tags: tuple[NodeTag, ...]
) -> None:
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    project = require_project(workspace, name)
    assert project["required_tags"] == tags
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path="tools/fleet-execution",
        install=(),
        companions=(),
    )
