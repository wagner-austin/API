"""The fairy's art pipeline runs on the fleet as fairy/art (MCPs board task 5211b20a).

fairy/art is the Python package in the fairy repository's art/ directory that
turns Gemini's 21:9 masters into the 1600 x 720 plates the fairy's
touchscreen shows, and verifies every committed asset against its record.
Its make check installs itself with poetry, imports mcp-shared-py by the
path ../../MCPs/mcp-shared-py and runs maketools from the same checkout, so
it stages MCPs beside the export as a companion. Its mutation step measures
what a commit changes against refs/remotes/origin/main, so it declares
base_branch main; without one the stage holds a single root commit and the
step has no base to measure against. Its Makefile takes zlib from the
nixpkgs input of the companion's fleet flake with nix build, since Pillow's
wheels link the system's libz.so.1 and a plate's PNG bytes depend on which
zlib wrote them; every enabled worker measured on 2026-11-23 (citrine,
colossus, diphtheria, lavender, lenovold, loki, pendragon, sedona and
serendipity) runs NixOS with nix on its PATH, so linux is the only tag. Its
282 tests peaked at 261 MB resident in one process on ziggurat (fairy
a0e1acc), so half a gigabyte holds a worker.
"""

from __future__ import annotations

from fleet.contracts.source import ProjectCompanion, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document


def test_the_art_check_stages_mcps_and_measures_against_main_on_any_linux_node() -> None:
    project = require_project(decode_fleet_workspace(shipped_document()), "fairy/art")

    assert project["required_tags"] == (NodeTag.LINUX,)
    assert project["node"] is None
    assert project["minimum_workers"] == 2
    assert project["worker_ram_gb"] == 0.5
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/fairy.git",
        path="art",
        install=(),
        companions=(
            ProjectCompanion(
                remote="https://github.com/wagner-austin/MCPs.git", ref="main", directory="MCPs"
            ),
        ),
        base_branch="main",
    )
