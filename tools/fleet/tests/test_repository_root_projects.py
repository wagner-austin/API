"""Each room's repository has its whole make check on the fleet (MCPs board task a8ee9b21).

The review pass gathers a PASSED check for every repository a closure
touches, and a repository whose check can run only on a workstation shows as
FAILED there. hardware-wiki's, metabolomics-dashboard's, chat's and
corvis-stick's checks run MCPs' published maketools, which reads ``../MCPs``
at ``origin/main``, so each
carries MCPs as a companion at ``main``: the fleet clones it beside the export
with ``origin/main`` at its real commit (fleet.core.dialect,
companion_repository_commands). idle's check reads nothing beside it, and its
browser project draws with WebGL in Chromium, so it takes slime's tags.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str

from fleet.contracts.source import InstallStep, ProjectCompanion, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project

#: The install step of every npm repository's root check.
NPM_CI = InstallStep(phase="install", argv=("npm", "ci"))

#: MCPs, staged beside a project whose check runs MCPs' published maketools.
MCPS_COMPANION = ProjectCompanion(
    remote="https://github.com/wagner-austin/MCPs.git", ref="main", directory="MCPs"
)


@pytest.mark.parametrize(
    ("name", "tags", "install", "companions"),
    [
        (
            "idle",
            (NodeTag.GPU, NodeTag.WINDOWS),
            (
                NPM_CI,
                InstallStep(phase="install", argv=("npx", "playwright", "install", "chromium")),
            ),
            (),
        ),
        ("chat", (NodeTag.WINDOWS,), (NPM_CI,), (MCPS_COMPANION,)),
        ("hardware-wiki", (), (), (MCPS_COMPANION,)),
        ("metabolomics-dashboard", (NodeTag.WINDOWS,), (), (MCPS_COMPANION,)),
        ("corvis-stick", (NodeTag.WINDOWS,), (NPM_CI,), (MCPS_COMPANION,)),
    ],
)
def test_each_repository_s_root_check_is_declared_with_what_it_reads(
    name: str,
    tags: tuple[NodeTag, ...],
    install: tuple[InstallStep, ...],
    companions: tuple[ProjectCompanion, ...],
) -> None:
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    project = require_project(workspace, name)
    assert project["required_tags"] == tags
    assert project["source"] == ProjectSource(
        remote=f"https://github.com/wagner-austin/{name}.git",
        path="",
        install=install,
        companions=companions,
    )
