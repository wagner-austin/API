"""The fleet checks MCPs packages/claude-hooks once per platform (MCPs board task c1832d51).

The hooks run on Windows on the hub and the fleet's Windows nodes, and on
Linux on ziggurat, where the operator's sessions run, and the package keeps a
real arm per platform in its suites. Until 2026-10-17 the registry declared
only the Windows project, so its check ran on sedona alone: on ziggurat the
package's make check stood at 99.40% coverage, every Linux-only line unrun,
and nothing said so. This decodes the real registry to require one project
per platform over the one directory, each carrying ``hooks`` (the route file
and the pinned tools the check runs on) and nothing that would isolate the
build from the account whose route file the live suites read.
"""

from __future__ import annotations

import pytest

from fleet.contracts.source import ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document


@pytest.mark.parametrize(
    ("name", "tags"),
    [
        ("MCPs/packages/claude-hooks", (NodeTag.WINDOWS, NodeTag.HOOKS)),
        ("MCPs/packages/claude-hooks-linux", (NodeTag.LINUX, NodeTag.HOOKS)),
    ],
)
def test_each_platform_checks_the_hooks_package(name: str, tags: tuple[NodeTag, ...]) -> None:
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, name)
    assert project["required_tags"] == tags
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/MCPs.git",
        path="packages/claude-hooks",
        install=(),
        companions=(),
    )
