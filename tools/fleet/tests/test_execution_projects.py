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

import pathlib
from typing import Final

import pytest

from fleet.cli.runners import load_runner_spec
from fleet.contracts.source import InstallStep, ProjectCompanion, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_node, require_project
from tests._host import RUNNER_HOST_OPTION
from tests._registry import shipped_document

#: This package, whose recipe and roster the runner host's suite reads.
PACKAGE: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent.parent


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
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, name)
    assert project["required_tags"] == tags
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path="tools/fleet-execution",
        install=(),
        companions=(),
    )


def test_the_runner_host_s_cases_run_on_its_node_as_the_fleet_user() -> None:
    """tools/fleet-execution-runners reads lavender's live runner units.

    It is pinned to the node whose WSL host is the one its recipe names to
    the cases, which ``runners.json`` declares, and it requires no ``docker``
    tag: that would isolate it as execdocker, while its full-slice case runs
    under the fleet user's own systemd manager (MCPs board task 715b9ec5).
    """
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, "tools/fleet-execution-runners")
    assert (project["required_tags"], project["node"]) == ((NodeTag.LINUX,), "lavender-wsl")
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path="tools/fleet-execution-runners",
        install=(),
        companions=(),
    )
    recipe = (PACKAGE / "Makefile").read_text(encoding="utf-8")
    given = f"{RUNNER_HOST_OPTION}="
    named = [word.removeprefix(given) for word in recipe.split() if word.startswith(given)]
    host = require_node(workspace, "lavender-wsl")["wsl_host"]
    declared = [each["name"] for each in load_runner_spec(str(PACKAGE / "runners.json"))["hosts"]]
    assert (named, host in declared) == ([host], True)


def test_the_deploy_cases_run_isolated_on_a_rootless_daemon() -> None:
    """tools/maketools' deploy cases reach only execdocker's daemon.

    Each case builds a service's image and starts and removes its container
    through the real compose path (board task 465689f5), so the project
    requires ``docker``, which makes the runner isolate it, and ``linux``,
    the only platform that tag is carried on.
    """
    workspace = decode_fleet_workspace(shipped_document())
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
    workspace = decode_fleet_workspace(shipped_document())
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
    workspace = decode_fleet_workspace(shipped_document())
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


@pytest.mark.parametrize("host", ["diphtheria", "colossus"])
def test_each_served_lane_runs_on_its_search_host_alone(host: str) -> None:
    """MCPs/execution-served-<host> reads what that host's search container serves.

    Its last install step writes the marker that makes the execution
    workspace collect tests/served alone (MCPs board tasks b456a3cb and
    bf7a9460), and its case reads 127.0.0.1:8090, which only that host's
    build reaches: a run on any other node would read nothing, and one run
    through the public tunnel reads whichever origin its node's route
    reaches. So each project names its host, and every other node refuses it.
    It holds no test database and no companion, so it takes the npm workspace
    build's tags and nothing more.
    """
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, f"MCPs/execution-served-{host}")
    assert project["node"] == host
    assert project["required_tags"] == (NodeTag.LINUX, NodeTag.CXX)
    assert project["exclusive_resources"] == ()
    prepare = ("make", "-s", "-C", "scripts/fleet-prepare")
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/MCPs.git",
        path="execution",
        install=(
            InstallStep(phase="install", argv=(*prepare, "install")),
            InstallStep(phase="workspace-build", argv=(*prepare, "workspace-build")),
            InstallStep(phase="served-lane", argv=(*prepare, "served-lane")),
        ),
        companions=(),
    )


def test_the_live_demo_case_runs_where_ffprobe_is() -> None:
    """clients/TankpitBot's live demo case lands on a Windows node with ffmpeg.

    The case reads a segment the public demo served with ``ffprobe`` to
    require its AAC audio stream (MCPs board task 46934cd6), so a node
    without the ``ffmpeg`` tag could only fail it; ``windows`` is the
    platform the package's own check is registered for.
    """
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, "clients/TankpitBot-execution")
    assert project["required_tags"] == (NodeTag.WINDOWS, NodeTag.FFMPEG)
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path="clients/TankpitBot-execution",
        install=(),
        companions=(),
    )
