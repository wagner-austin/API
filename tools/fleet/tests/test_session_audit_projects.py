"""The fleet checks MCPs packages/session-audit on Linux alone (MCPs board task c8cb0ac5).

Until 2026-10-17 the registry declared only a Windows project for the
package, so a Linux-only defect reached main unseen: on ziggurat its check
failed 79 cases and stood below 100% coverage while every fleet verdict
read green (MCPs board task 31e0a1eb), and a Linux lane was added beside
it. On 2026-10-21 the package dropped its Windows code ("well covnert any
windows code to linux no back compat", the operator, 2026-10-20), so the
Windows lane would check code that no longer exists. This decodes the real
registry to require exactly one project over the directory, on a Linux
node with the test database and the compiler its suites need.
"""

from __future__ import annotations

from typing import Final

from fleet.contracts.source import InstallStep, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document

#: The prepare recipe every MCPs workspace install goes through.
PREPARE: Final[tuple[str, ...]] = ("make", "-s", "-C", "scripts/fleet-prepare")

#: The source the lane checks: the package, its workspace installed and
#: built, and the fleet test database migrated.
SOURCE: Final[ProjectSource] = ProjectSource(
    remote="https://github.com/wagner-austin/MCPs.git",
    path="packages/session-audit",
    install=(
        InstallStep(phase="install", argv=(*PREPARE, "install")),
        InstallStep(phase="workspace-build", argv=(*PREPARE, "workspace-build")),
        InstallStep(
            phase="test-database",
            argv=(*PREPARE, "test-database", "CONTAINER=corvis-fleet-testdb"),
        ),
    ),
    companions=(),
)


def test_one_linux_project_checks_the_session_audit_package() -> None:
    workspace = decode_fleet_workspace(shipped_document())
    checking = sorted(
        name for name, project in workspace["projects"].items() if project["source"] == SOURCE
    )
    assert checking == ["MCPs/packages/session-audit-linux"]
    project = require_project(workspace, "MCPs/packages/session-audit-linux")
    assert project["required_tags"] == (NodeTag.LINUX, NodeTag.TESTDB, NodeTag.CXX)
    assert project["exclusive_resources"] == ("corvis-fleet-testdb",)
