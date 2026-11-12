"""coax-drone's site check runs on the fleet as coax-drone/site (MCPs board task c88f1f08).

The site in coax-drone's site/ directory shows the CAD export, its checks and
the project wiki at lift.austinwagner.org. Its make check installs its own
pinned TypeScript from site/package-lock.json, type-checks the site's modules
and runs tests/site under node's built-in runner and coverage, reading the
repository's committed exports and wiki from the same clone. So it declares no
install step and no companion, and needs only a Linux node: its recipe is
POSIX shell, and every node measured on 2026-11-12 (sedona, colossus,
lavender, demure, serendipity and pendragon) runs node v24.22.1, past the
22.8 that node's coverage thresholds need. It runs 45 tests, which finish in
seconds, so one worker of half a gigabyte holds it.
"""

from __future__ import annotations

from fleet.contracts.source import ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document


def test_the_site_check_runs_from_a_clean_coax_drone_export_on_any_linux_node() -> None:
    project = require_project(decode_fleet_workspace(shipped_document()), "coax-drone/site")

    assert project["required_tags"] == (NodeTag.LINUX,)
    assert project["node"] is None
    assert project["minimum_workers"] == 1
    assert project["worker_ram_gb"] == 0.5
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/coax-drone.git",
        path="site",
        install=(),
        companions=(),
    )
