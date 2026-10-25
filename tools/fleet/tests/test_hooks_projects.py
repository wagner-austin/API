"""The fleet checks MCPs packages/claude-hooks on Linux alone (MCPs board tasks c1832d51, 5b040fdb).

The hooks run where the operator's sessions run, ziggurat, which is Linux.
From 2026-10-17 the registry declared one project per platform over the one
directory, because the hooks also ran on the Windows hub and the fleet's
Windows nodes; on 2026-10-25 the operator ruled "so no windows shit then
ok", and the Windows project left the registry. This decodes the real
registry to require exactly one project over the directory, on Linux,
carrying ``hooks`` (the route file and the pinned tools the check runs on)
and nothing that would isolate the build from the account whose route file
the live suites read.
"""

from __future__ import annotations

from fleet.contracts.source import ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace
from tests._registry import shipped_document


def test_linux_alone_checks_the_hooks_package() -> None:
    workspace = decode_fleet_workspace(shipped_document())
    hooks = {
        name: project
        for name, project in workspace["projects"].items()
        if project["source"]["path"] == "packages/claude-hooks"
        and project["source"]["remote"] == "https://github.com/wagner-austin/MCPs.git"
    }
    assert sorted(hooks) == ["MCPs/packages/claude-hooks-linux"]
    project = hooks["MCPs/packages/claude-hooks-linux"]
    assert project["required_tags"] == (NodeTag.LINUX, NodeTag.HOOKS)
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/MCPs.git",
        path="packages/claude-hooks",
        install=(),
        companions=(),
    )
