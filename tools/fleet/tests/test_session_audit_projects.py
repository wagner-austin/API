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

import pytest

from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document


@pytest.mark.parametrize(
    ("name", "platform"),
    [
        ("MCPs/packages/session-audit", NodeTag.WINDOWS),
        ("MCPs/packages/session-audit-linux", NodeTag.LINUX),
    ],
)
def test_each_platform_checks_the_session_audit_package(name: str, platform: NodeTag) -> None:
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, name)
    windows = require_project(workspace, "MCPs/packages/session-audit")
    assert project["required_tags"] == (platform, NodeTag.TESTDB, NodeTag.CXX)
    source = project["source"]
    assert source is not None
    assert source["path"] == "packages/session-audit"
    assert source == windows["source"]
    assert project["exclusive_resources"] == windows["exclusive_resources"]
