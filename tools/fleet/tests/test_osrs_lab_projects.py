"""osrs-lab's two Gradle builds run on the fleet's docker lane (board task fd8e5f9b).

No fleet node has a JDK, Gradle or an X server: measured over ssh on
2026-11-02, colossus, demure and lavender each answered none of java,
gradle, xvfb-run or openbox, and each has docker. So each directory's
``make check`` builds the image its desktop-input/Dockerfile pins (JDK 17,
Gradle 8.10.2, Xvfb, Openbox) and runs Gradle inside it, which the
``docker`` tag sends to execdocker's rootless daemon. Run that way on
colossus, desktop-input passed 68 tests at lines 592/592 and branches
378/378, and the observer ran 477 tests to its own coverage gate. Neither
names a node: any node carrying the two tags may take it.
"""

from __future__ import annotations

import pytest

from fleet.contracts.source import ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document


@pytest.mark.parametrize("directory", ["desktop-input", "observer"])
def test_each_osrs_lab_build_runs_isolated_on_a_rootless_daemon(directory: str) -> None:
    """Each build reaches only execdocker's daemon, from a clean osrs-lab export.

    Both include their sibling modules (input-policy, brain, recorder) as
    Gradle subprojects of the same export, and the image is built inside
    the check, so the source declares no install step and no companion.
    """
    project = require_project(decode_fleet_workspace(shipped_document()), f"osrs-lab/{directory}")

    assert project["required_tags"] == (NodeTag.LINUX, NodeTag.DOCKER)
    assert project["node"] is None
    assert project["loopback_only"] is False
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/osrs-lab.git",
        path=directory,
        install=(),
        companions=(),
    )


@pytest.mark.parametrize("directory", ["desktop-input", "observer"])
def test_each_osrs_lab_build_holds_the_four_cores_and_six_gb_it_was_measured_in(
    directory: str,
) -> None:
    """Four workers of 1.5 GB are the 4 cores and 6 GB the colossus runs had."""
    project = require_project(decode_fleet_workspace(shipped_document()), f"osrs-lab/{directory}")

    assert project["minimum_workers"] == 4
    assert project["worker_ram_gb"] * project["minimum_workers"] == 6.0
