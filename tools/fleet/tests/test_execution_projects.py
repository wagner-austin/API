"""The fleet runs tools/fleet's host cases once per platform (board task 465689f5).

API's deploy refuses a commit unless the fleet queue records a passed run of
tools/fleet-execution and of tools/fleet-execution-linux at it (the root
Makefile's ``executed``, pinned in tools/maketools' execution-gate wiring
test). Those verdicts mean both halves ran only if the registry sends each
project to a node of its platform and both stage the one directory whose
check runs the host cases, which is what this decodes the real registry to
require.

The Linux half also requires ``docker``: its host cases include one that runs
a docker project's build as the node's execdocker user against that user's
rootless daemon (MCPs board task a8ee9b21), which only a node carrying the
tag has. slime/execution and idle/execution require it for the same
reason: each builds and starts its game's containers.

MCPs/execution-testdb carries MCPs itself as a companion: its pushed-tests
case replays a 2026-09-28 push, whose commits a node can read only from a
clone the hub sent beside the export (MCPs board task 93d7d7f4).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str

from fleet.contracts.source import InstallStep, ProjectCompanion, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project


@pytest.mark.parametrize(
    ("name", "tags"),
    [
        ("tools/fleet-execution", (NodeTag.WINDOWS,)),
        ("tools/fleet-execution-linux", (NodeTag.LINUX, NodeTag.DOCKER)),
    ],
)
def test_each_half_runs_the_execution_directory_on_its_platform(
    name: str, tags: tuple[NodeTag, ...]
) -> None:
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    project = require_project(workspace, name)
    assert project["required_tags"] == tags
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path="tools/fleet-execution",
        install=(),
        companions=(),
    )


def test_the_deploy_cases_run_isolated_on_a_rootless_daemon() -> None:
    """tools/maketools' deploy cases reach only execdocker's daemon.

    Each case builds a service's image and starts and removes its container
    through the real compose path (board task 465689f5), so the project
    requires ``docker``, which makes the runner isolate it, and ``linux``,
    the only platform that tag is carried on.
    """
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    project = require_project(workspace, "tools/maketools-execution")
    assert project["required_tags"] == (NodeTag.LINUX, NodeTag.DOCKER)
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path="tools/maketools-execution",
        install=(),
        companions=(),
    )


@pytest.mark.parametrize("game", ["slime", "idle"])
def test_each_games_suite_runs_isolated_on_a_rootless_daemon(game: str) -> None:
    """A game's make up refuses a commit <game>/execution has not passed at.

    Each suite builds, starts and deletes containers, so it must only ever
    reach execdocker's rootless daemon: the ``docker`` tag is what makes
    the runner isolate it (dispatch.recipe_for), and each game's own script
    refuses any other daemon besides (MCPs board task a8ee9b21).
    """
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    project = require_project(workspace, f"{game}/execution")
    assert project["required_tags"] == (NodeTag.LINUX, NodeTag.DOCKER)
    assert project["source"] == ProjectSource(
        remote=f"https://github.com/wagner-austin/{game}.git",
        path="execution",
        install=(InstallStep(phase="install", argv=("npm", "ci")),),
        companions=(),
    )


def test_the_testdb_lane_reads_the_pushed_commits_from_an_mcps_companion() -> None:
    """MCPs/execution-testdb clones MCPs' history to ``../MCPs``.

    Its pushed-tests case fetches 53274ebe9, bbee12d4b, 49b8cb6a1 and
    ea4e20239 to replay the push pre-push now refuses. The node holds no git
    credential, and job 9f8b9628 on lavender-wsl failed at that fetch from
    GitHub with "could not read Username"; the companion is the hub's
    full-history bundle of ``main``, which holds all four as ancestors, at
    the path a workstation's worktree reaches the main checkout by
    (MCPs board task 93d7d7f4). The lane holds the shared test database
    exclusively, since the replayed suite runs against ``corvis_test``.
    """
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    project = require_project(workspace, "MCPs/execution-testdb")
    assert project["required_tags"] == (NodeTag.LINUX, NodeTag.TESTDB, NodeTag.CXX)
    assert project["exclusive_resources"] == ("corvis-fleet-testdb",)
    prepare = ("make", "-s", "-C", "scripts/fleet-prepare")
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/MCPs.git",
        path="execution",
        install=(
            InstallStep(phase="install", argv=(*prepare, "install")),
            InstallStep(phase="workspace-build", argv=(*prepare, "workspace-build")),
            InstallStep(
                phase="test-database",
                argv=(*prepare, "test-database", "CONTAINER=corvis-fleet-testdb"),
            ),
        ),
        companions=(
            ProjectCompanion(
                remote="https://github.com/wagner-austin/MCPs.git", ref="main", directory="MCPs"
            ),
        ),
    )


def test_the_live_demo_case_runs_where_ffprobe_is() -> None:
    """clients/TankpitBot's live demo case lands on a Windows node with ffmpeg.

    The case reads a segment the public demo served with ``ffprobe`` to
    require its AAC audio stream (MCPs board task 46934cd6), so a node
    without the ``ffmpeg`` tag could only fail it; ``windows`` is the
    platform the package's own check is registered for.
    """
    path = Path(__file__).resolve().parents[1] / "fleet.json"
    workspace = decode_fleet_workspace(load_json_str(path.read_text(encoding="utf-8")))
    project = require_project(workspace, "clients/TankpitBot-execution")
    assert project["required_tags"] == (NodeTag.WINDOWS, NodeTag.FFMPEG)
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path="clients/TankpitBot-execution",
        install=(),
        companions=(),
    )
