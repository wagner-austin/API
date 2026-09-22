"""The registered slime build is self-contained, including its integrity gate."""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str

from fleet.contracts.source import ProjectSource
from fleet.contracts.workspace import decode_fleet_workspace, require_project


def test_slime_does_not_stage_an_unused_source_repository() -> None:
    """Decode the real registry and require only slime's own source and toolchain."""
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    document = load_json_str(path.read_text(encoding="utf-8"))
    workspace = decode_fleet_workspace(document)
    project = require_project(workspace, "slime")
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/slime.git",
        path="",
        install=(("npm", "ci"), ("npx", "playwright", "install", "chromium", "webkit")),
        companions=(),
    )
