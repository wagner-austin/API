"""The fleet runs tools/fleet's host cases once per platform (board task 465689f5).

API's deploy refuses a commit unless the fleet queue records a passed run of
tools/fleet-execution and of tools/fleet-execution-linux at it (the root
Makefile's ``executed``, pinned in tools/maketools' execution-gate wiring
test). Those verdicts mean both halves ran only if the registry sends each
project to a node of its platform and both stage the one directory whose
check runs the host cases, which is what this decodes the real registry to
require.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str

from fleet.contracts.source import ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project


@pytest.mark.parametrize(
    ("name", "tag"),
    [
        ("tools/fleet-execution", NodeTag.WINDOWS),
        ("tools/fleet-execution-linux", NodeTag.LINUX),
    ],
)
def test_each_half_runs_the_execution_directory_on_its_platform(name: str, tag: NodeTag) -> None:
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    project = require_project(workspace, name)
    assert project["required_tags"] == (tag,)
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path="tools/fleet-execution",
        install=(),
        companions=(),
    )
