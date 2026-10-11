"""The registered slime build is self-contained, including its integrity gate."""

from __future__ import annotations

from fleet.contracts.source import InstallStep, ProjectSource
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document


def test_slime_does_not_stage_an_unused_source_repository() -> None:
    """Decode the real registry and require only slime's own source and toolchain."""
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, "slime")
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/slime.git",
        path="",
        install=(
            InstallStep(phase="install", argv=("npm", "ci")),
            InstallStep(
                phase="install", argv=("npx", "playwright", "install", "chromium", "webkit")
            ),
        ),
        companions=(),
    )
