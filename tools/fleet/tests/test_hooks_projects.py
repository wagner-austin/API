"""The fleet checks MCPs packages/claude-hooks on Linux alone (MCPs board tasks c1832d51, 5b040fdb).

The hooks run where the operator's sessions run, ziggurat, which is Linux.
From 2026-10-17 the registry declared one project per platform over the one
directory, because the hooks also ran on the Windows hub and the fleet's
Windows nodes; on 2026-10-25 the operator ruled "so no windows shit then
ok", and the Windows project left the registry. This decodes the real
registry to require exactly one project over the directory, on Linux,
carrying ``hooks`` (the route file and the pinned tools the check runs on)
and nothing that would isolate the build from the account whose route file
the live suites read. It declares ``base_branch`` main (MCPs board task
9c456c33): the package's check runs the mutation step, which measures what
the commit adds against ``refs/remotes/origin/main``, and fleet job
cda32861 refused it ``MUTATION_BASE_UNAVAILABLE`` while the entry declared
none, so the stage held one root commit.

SINCE MCPs BOARD TASK c35a9ac3 the check also needs a disposable board:
tests/test_board_wake_live.py runs board-wake.py's main over the real
client against taskboard-mcp behind a real mcp-proxy, both from the staged
tree's build, on the node's freshly cloned ``corvis_test``. So the entry
installs the workspace, builds ``packages/db``, ``taskboard-mcp`` and
``mcp-proxy`` with their dependencies, and readies the test database, as
the session-audit entry does for its audit world. It requires ``testdb``
and ``cxx`` beside ``hooks``, and holds ``corvis-fleet-testdb`` alone while
it runs, since the database is the node's one test cluster's.
"""

from __future__ import annotations

from fleet.contracts.source import InstallStep, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document

MCPS_REMOTE = "https://github.com/wagner-austin/MCPs.git"

#: The commands the entry runs at the export root, in order, before the check.
PREPARE = ("make", "-s", "-C", "scripts/fleet-prepare")


def test_linux_alone_checks_the_hooks_package() -> None:
    workspace = decode_fleet_workspace(shipped_document())
    hooks = sorted(
        name
        for name, project in workspace["projects"].items()
        if (source := project["source"]) is not None
        and source["path"] == "packages/claude-hooks"
        and source["remote"] == MCPS_REMOTE
    )
    assert hooks == ["MCPs/packages/claude-hooks-linux"]
    project = require_project(workspace, "MCPs/packages/claude-hooks-linux")
    assert project["required_tags"] == (
        NodeTag.LINUX,
        NodeTag.HOOKS,
        NodeTag.TESTDB,
        NodeTag.CXX,
    )
    assert project["exclusive_resources"] == ("corvis-fleet-testdb",)
    assert project["source"] == ProjectSource(
        remote=MCPS_REMOTE,
        path="packages/claude-hooks",
        install=(
            InstallStep(phase="install", argv=(*PREPARE, "install")),
            InstallStep(
                phase="workspace-build",
                argv=(*PREPARE, "workspace-build", "ROOTS=packages/db:taskboard-mcp:mcp-proxy"),
            ),
            InstallStep(
                phase="test-database",
                argv=(*PREPARE, "test-database", "CONTAINER=corvis-fleet-testdb"),
            ),
        ),
        companions=(),
        base_branch="main",
    )
