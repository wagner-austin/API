"""The fleet checks MCPs packages/session-audit once per platform (MCPs board task 31e0a1eb).

session-audit runs on the Windows hub and, since 2026-10-16, on ziggurat, the
Linux session host the operator's sessions moved to, and its suites keep a
real arm per platform. Until 2026-10-17 the registry declared only the
Windows project, so a Linux-only defect reached main unseen: on ziggurat the
package's check failed 79 cases and stood below 100% coverage while every
fleet verdict read green. This decodes the real registry to require one
project per platform over the one directory, each with the test database
and the compiler its suites need and the same install, so the two lanes
check the same build.
"""

from __future__ import annotations

from typing import Final

import pytest

from fleet.contracts.source import InstallStep, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document

#: The prepare recipe every MCPs workspace install goes through.
PREPARE: Final[tuple[str, ...]] = ("make", "-s", "-C", "scripts/fleet-prepare")

#: The source both lanes check: the package, its workspace installed and
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


@pytest.mark.parametrize(
    ("name", "platform"),
    [
        ("MCPs/packages/session-audit", NodeTag.WINDOWS),
        ("MCPs/packages/session-audit-linux", NodeTag.LINUX),
    ],
)
def test_each_platform_checks_the_session_audit_package(name: str, platform: NodeTag) -> None:
    project = require_project(decode_fleet_workspace(shipped_document()), name)
    assert project["required_tags"] == (platform, NodeTag.TESTDB, NodeTag.CXX)
    assert project["source"] == SOURCE
    assert project["exclusive_resources"] == ("corvis-fleet-testdb",)
