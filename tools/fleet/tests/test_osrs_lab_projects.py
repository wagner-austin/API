"""osrs-lab's Gradle builds run on the fleet's docker lane (board task fd8e5f9b).

No fleet node has a JDK, Gradle or an X server: measured over ssh on
2026-11-02, colossus, demure and lavender each answered none of java,
gradle, xvfb-run or openbox, and each has docker. So each directory's
``make check`` builds the image its desktop-input/Dockerfile pins (JDK 17,
Gradle 8.10.2, Xvfb, Openbox) and runs Gradle inside it, which the
``docker`` tag sends to execdocker's rootless daemon. Run that way on
colossus, desktop-input passed 68 tests at lines 592/592 and branches
378/378, and the observer ran 477 tests to its own coverage gate. Neither
names a node: any node carrying the two tags may take it.

observer-input-coverage (board task 4a83389f) runs the same observer build,
then only its own per-class JaCoCo gate over the observer's input classes
that read no live Client, so its COVERAGE TOTAL is theirs alone: on
colossus 479 tests passed at lines 142/142 and branches 66/66.

recorder (board task bc97ff15) runs the passive recorder's own make check in
the same image, whose release gate merges the recorded logged-in coverage of
its live readers: on colossus 359 tests passed at lines 1723/1723 and
branches 1453/1453. recorder-release runs the same build through its
packaging gate, then starts the jar it packed accountless and requires that
jar's own complete audit of the trace it writes.

brain (board task 488e5f06) runs the planner library's own make check in the
same image since osrs-lab 4fd53f3, with the manual-play replay corpus it
holds: on colossus 267 tests passed at lines 1112/1112 and branches 1109/1109.

live-coverage (board task 488e5f06) runs the shared check of kept JaCoCo live
runs, which the recorder and observer gates both execute, in the same image:
on colossus it passed with input-policy, 13 tests at lines 85/85 and branches
60/60.

manual-desktop (board task b2ee9cbe) runs the manual desktop image's
configuration tests in the same image, starting real PulseAudio daemons, beside
input-policy's: on colossus at osrs-lab e33ff2b it passed 9 tests, 4 of its
own and 5 of input-policy's, in 30 s, at lines 37/37 and branches 26/26, all
input-policy's, since the directory holds no main source.

mobile (MCPs board task 0ca06475) runs the official Android client's USB link,
H.264 decoder and HUD locator's own make check in the same image, with JavaCPP's
FFmpeg natives from Maven and real Pixel 8 recordings as fixtures: run with the
same Makefile on diphtheria at osrs-lab 4e1a6837 it passed 82 tests, 75 of its
own and 7 of input-policy's, at lines 592/592 and branches 278/278.

input-policy (board task b2ee9cbe) runs the input boundary policy's own make
check in the same image since osrs-lab 6176e3d, whose packaged-classes.gradle
the recorder and the observer's login plugin artifact apply: on colossus it
passed 5 tests in 18 s at lines 37/37 and branches 26/26.

phone-run (MCPs board task ea59fdca) is not a suite but a live run of the
phone bot against the Pixel 8 on diphtheria's USB, and the queue is the only
way one starts: on 2026-11-14 osrs-lab containers started by hand on
diphtheria shared its four cores with doc-extract-api's check, which passed
every test and overran its budget three times (fleet jobs 43d0f554, 926a22d4
and 13af2e26). Its make check builds mobile's classes in the same image, then
runs the run's main on the host, where adb's server and the phone are, so it
is pinned there, holds the device lease MCPs' phone lanes hold, and is
staged on main's history, from which it names the commit it runs.

accounts (MCPs board task 0bd4fb82) is not a Gradle build but the Python
registry of every character the bots or the operator play and the report of
their play, held to the shared guards, strict mypy and 100% coverage as
osrs-lab's evaluation package is. Its mcp-shared-py dependency is MCPs'
checkout beside the export, so MCPs is its companion, and the guard imports
that package's contract_hash.py, which MCPs generates and does not commit,
so its own make install generates it before installing the venv, as
coax-drone's freecad-environment does: an install step's argv is rendered
into the node's shell and may not open with ``../``, and no CUDA wheel in
its lock calls for a deps phase. It runs no container, so it needs only a
linux node.
"""

from __future__ import annotations

from typing import Final

import pytest

from fleet.contracts.release_artifact import ArtifactKind, ReleaseArtifact
from fleet.contracts.source import InstallStep, ProjectCompanion, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document

DIRECTORIES: Final[tuple[str, ...]] = (
    "brain",
    "build-runner",
    "desktop-input",
    "input-policy",
    "live-coverage",
    "manual-desktop",
    "mobile",
    "observer",
    "observer-input-coverage",
    "recorder",
    "recorder-release",
)


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_each_osrs_lab_build_runs_isolated_on_a_rootless_daemon(directory: str) -> None:
    """Each build reaches only execdocker's daemon, from a clean osrs-lab export.

    Each includes its sibling modules (input-policy, brain, recorder) as
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


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_each_osrs_lab_build_holds_the_four_cores_and_six_gb_it_was_measured_in(
    directory: str,
) -> None:
    """Four workers of 1.5 GB are the 4 cores and 6 GB the colossus runs had."""
    project = require_project(decode_fleet_workspace(shipped_document()), f"osrs-lab/{directory}")

    assert project["minimum_workers"] == 4
    assert project["worker_ram_gb"] * project["minimum_workers"] == 6.0


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_only_the_release_gate_declares_the_jar_it_packs(directory: str) -> None:
    """recorder-release's packaging gate leaves the recorder jar loki runs.

    Its verdict names that jar by its entries (board task 0c8ace8d); every
    other build releases nothing.
    """
    project = require_project(decode_fleet_workspace(shipped_document()), f"osrs-lab/{directory}")

    expected = (
        ReleaseArtifact(path="recorder/build/libs/osrs-recorder.jar", kind=ArtifactKind.ZIP)
        if directory == "recorder-release"
        else None
    )
    assert project["release_artifact"] == expected


def test_the_phone_run_is_pinned_to_the_pixel_s_host_and_holds_its_device_lease() -> None:
    """The live run goes only to diphtheria, one at a time with the phone lanes.

    Its build needs execdocker's daemon as every osrs-lab build does, its main
    needs the host's adb server, which only diphtheria's USB reaches, and it
    shares MCPs' phone lanes' lease. It is staged on main's history, so its
    tree's parent is the commit it names in its trace.
    """
    project = require_project(decode_fleet_workspace(shipped_document()), "osrs-lab/phone-run")

    assert project["node"] == "diphtheria"
    assert project["required_tags"] == (NodeTag.LINUX, NodeTag.DOCKER)
    assert project["exclusive_resources"] == ("corvis-phone-device-tests",)
    assert project["loopback_only"] is False
    assert project["release_artifact"] is None
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/osrs-lab.git",
        path="phone-run",
        install=(),
        companions=(),
        base_branch="main",
    )


def test_the_accounts_registry_installs_beside_mcps_on_any_linux_node() -> None:
    """Its own make install generates mcp-shared-py's hash beside MCPs, then installs its venv."""
    project = require_project(decode_fleet_workspace(shipped_document()), "osrs-lab/accounts")

    assert project["required_tags"] == (NodeTag.LINUX,)
    assert project["node"] is None
    assert project["exclusive_resources"] == ()
    assert project["release_artifact"] is None
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/osrs-lab.git",
        path="accounts",
        install=(InstallStep(phase="install", argv=("make", "-C", "accounts", "install")),),
        companions=(
            ProjectCompanion(
                remote="https://github.com/wagner-austin/MCPs.git", ref="main", directory="MCPs"
            ),
        ),
    )
